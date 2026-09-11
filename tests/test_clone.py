"""Prompt wrapping and official AutoModel load helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_clone import load_cosyvoice, prompt_length_warning, wrap_instruct_text, wrap_prompt_text


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


class LoadCosyvoiceTest(unittest.TestCase):
    def test_loads_official_automodel_once(self):
        created = []
        dirs = []

        def fake_auto_model(**kwargs):
            obj = object()
            created.append(obj)
            dirs.append(kwargs["model_dir"])
            return obj

        model = load_cosyvoice(
            Path("pretrained_models/Fun-CosyVoice3-0.5B"),
            fp16=True,
            AutoModel=fake_auto_model,
        )
        self.assertEqual(len(created), 1)
        self.assertIs(model, created[0])
        self.assertIn("Fun-CosyVoice3-0.5B", dirs[0].replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
