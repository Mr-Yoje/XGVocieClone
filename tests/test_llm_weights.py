"""Talker checkpoint loading must match official CosyVoice3 (fp32 weights + autocast)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_clone import coerce_llm_state_dict, prompt_length_warning, wrap_instruct_text, wrap_prompt_text


class FakeTensor(dict):
    """Minimal stand-in so we do not require torch in this unit test."""


class CoerceStateDictTest(unittest.TestCase):
    def test_keeps_raw_weight_dict(self):
        raw = {
            "llm.model.embed_tokens.weight": object(),
            "speech_embedding.weight": object(),
            "llm_decoder.weight": object(),
        }
        self.assertIs(coerce_llm_state_dict(raw), raw)

    def test_unwraps_training_checkpoint_wrapper(self):
        inner = {"speech_embedding.weight": object()}
        wrapped = {"epoch": 1, "state_dict": inner}
        self.assertEqual(coerce_llm_state_dict(wrapped), inner)


class ApplyLlmWeightsTest(unittest.TestCase):
    def test_does_not_cast_llm_to_half_when_fp16_flag_set(self):
        import demo_clone

        llm = MagicMock()
        llm.load_state_dict.return_value = ([], [])
        model = types.SimpleNamespace(device="cpu", fp16=True, llm=llm)
        cosy = types.SimpleNamespace(model=model)
        ckpt = Path("llm.rl.pt")
        fake_state = {"speech_embedding.weight": object()}

        orig_torch = sys.modules.get("torch")
        torch_mod = types.ModuleType("torch")
        torch_mod.load = MagicMock(return_value=fake_state)
        sys.modules["torch"] = torch_mod
        try:
            demo_clone.apply_llm_weights(cosy, ckpt, label="RL")
        finally:
            if orig_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = orig_torch

        llm.half.assert_not_called()
        llm.float.assert_not_called()
        llm.load_state_dict.assert_called_once()
        kwargs = llm.load_state_dict.call_args
        self.assertEqual(kwargs.args[0], fake_state)


class PromptWrapTest(unittest.TestCase):
    def test_zero_shot_prompt_keeps_endofprompt_before_transcript(self):
        wrapped = wrap_prompt_text("希望你以后能够做的比我还好呦。")
        self.assertTrue(wrapped.startswith("You are a helpful assistant.<|endofprompt|>"))
        self.assertIn("希望你以后能够做的比我还好呦。", wrapped)

    def test_instruct_puts_style_before_endofprompt(self):
        wrapped = wrap_instruct_text("请用开心的语气说")
        self.assertEqual(
            wrapped,
            "You are a helpful assistant. 请用开心的语气说<|endofprompt|>",
        )

    def test_warns_when_tts_much_shorter_than_prompt(self):
        msg = prompt_length_warning("短句。", "这是一段非常长的参考音频转写，" * 8)
        self.assertIsNotNone(msg)
        self.assertIn("过短", msg)


class IndependentModelsTest(unittest.TestCase):
    def _fake_model_dir(self):
        import tempfile

        td = Path(tempfile.mkdtemp())
        (td / "llm.pt").write_bytes(b"base")
        (td / "llm.rl.pt").write_bytes(b"rl")
        (td / "cosyvoice3.yaml").write_text("x", encoding="utf-8")
        return td

    def test_loads_two_separate_cosyvoice_instances(self):
        from demo_clone import load_independent_cosyvoices

        td = self._fake_model_dir()
        created = []
        dirs = []

        def fake_auto_model(**kwargs):
            obj = object()
            created.append(obj)
            dirs.append(Path(kwargs["model_dir"]))
            return obj

        models, shared = load_independent_cosyvoices(
            td,
            fp16=True,
            AutoModel=fake_auto_model,
            apply_rl=lambda *a, **k: self.fail("RL 应走官方 llm.pt 加载，不应再灌权重"),
            rl_ckpt=td / "llm.rl.pt",
        )
        self.assertFalse(shared)
        self.assertEqual(len(created), 2)
        self.assertIsNot(models["base"], models["rl"])
        self.assertEqual(dirs[0].resolve(), td.resolve())
        self.assertEqual((dirs[1] / "llm.pt").read_bytes(), b"rl")

    def test_shared_talker_reuses_one_instance(self):
        from demo_clone import load_independent_cosyvoices

        td = self._fake_model_dir()

        def fake_auto_model(**kwargs):
            return object()

        models, shared = load_independent_cosyvoices(
            td,
            fp16=False,
            shared_talker=True,
            AutoModel=fake_auto_model,
            apply_rl=lambda *a, **k: self.fail("should not apply RL at load"),
            rl_ckpt=td / "llm.rl.pt",
        )
        self.assertTrue(shared)
        self.assertIs(models["base"], models["rl"])


if __name__ == "__main__":
    unittest.main()
