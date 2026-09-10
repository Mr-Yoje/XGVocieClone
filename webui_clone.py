"""Gradio UI for CosyVoice 3-0.5B-RL voice cloning, with base vs RL compare."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from demo_clone import (
    add_cosyvoice_to_path,
    apply_llm_weights,
    concatenate_speech,
    llm_checkpoint_path,
    maybe_force_cpu,
    wrap_prompt_text,
)


DEFAULT_TEXT = "你好，这是一次 Fun-CosyVoice 3 零样本声音克隆测试。希望你能听出和参考音频相近的音色。"
MODE_COMPARE = "对比 RL 与基座"
MODE_RL = "仅 RL (llm.rl.pt)"
MODE_BASE = "仅基座 (llm.pt)"


class CloneEngine:
    def __init__(self, cosyvoice_root: Path, model_dir: Path, fp16: bool, force_gpu: bool):
        add_cosyvoice_to_path(cosyvoice_root)
        maybe_force_cpu(force_gpu)
        from cosyvoice.cli.cosyvoice import AutoModel

        if not (model_dir / "cosyvoice3.yaml").exists():
            raise FileNotFoundError(f"模型目录无效: {model_dir}")
        self.model_dir = model_dir
        self.base_ckpt = llm_checkpoint_path(model_dir, "base")
        self.rl_ckpt = llm_checkpoint_path(model_dir, "rl")
        print(f"加载模型: {model_dir}")
        t0 = time.time()
        self.model = AutoModel(model_dir=str(model_dir), load_trt=False, fp16=fp16)
        # AutoModel 默认 llm.pt
        self.active = "base"
        print(f"模型就绪 {time.time() - t0:.1f}s（当前 talker=基座）")

    def switch_talker(self, variant: str) -> None:
        if variant == self.active:
            return
        path = self.rl_ckpt if variant == "rl" else self.base_ckpt
        apply_llm_weights(self.model, path, label="RL" if variant == "rl" else "基座")
        self.active = variant

    def _infer(self, prompt_wav: str, prompt_text: str, tts_text: str, instruct: str, speed: float):
        chunks = []
        if instruct and instruct.strip():
            for item in self.model.inference_instruct2(
                tts_text.strip(),
                wrap_prompt_text(instruct.strip()),
                prompt_wav,
                stream=False,
                speed=float(speed),
            ):
                chunks.append(item)
        else:
            for item in self.model.inference_zero_shot(
                tts_text.strip(),
                wrap_prompt_text(prompt_text),
                prompt_wav,
                stream=False,
                speed=float(speed),
            ):
                chunks.append(item)
        if not chunks:
            raise RuntimeError("推理没有返回音频。")
        return concatenate_speech(chunks)

    def _save(self, speech, tag: str) -> tuple[str, float]:
        import torchaudio

        out = Path(tempfile.gettempdir()) / f"cosyvoice3_{tag}_{int(time.time() * 1000)}.wav"
        torchaudio.save(str(out), speech, self.model.sample_rate)
        dur = speech.shape[1] / self.model.sample_rate
        return str(out), dur

    def synthesize(self, prompt_audio, prompt_text: str, tts_text: str, instruct: str, speed: float, mode: str):
        if prompt_audio is None:
            raise ValueError("请先上传或录制参考音频。")
        if not prompt_text or not prompt_text.strip():
            raise ValueError("请填写参考音频的逐字转写。")
        if not tts_text or not tts_text.strip():
            raise ValueError("请填写要合成的文本。")

        want_rl = mode in (MODE_COMPARE, MODE_RL)
        want_base = mode in (MODE_COMPARE, MODE_BASE)
        lines = []
        rl_path = None
        base_path = None

        if want_base:
            self.switch_talker("base")
            t0 = time.time()
            speech = self._infer(prompt_audio, prompt_text, tts_text, instruct, speed)
            base_path, dur = self._save(speech, "base")
            elapsed = time.time() - t0
            lines.append(
                f"基座 llm.pt：时长 {dur:.2f}s · 用时 {elapsed:.2f}s · RTF {elapsed / max(dur, 1e-6):.3f}"
            )

        if want_rl:
            self.switch_talker("rl")
            t0 = time.time()
            speech = self._infer(prompt_audio, prompt_text, tts_text, instruct, speed)
            rl_path, dur = self._save(speech, "rl")
            elapsed = time.time() - t0
            lines.append(
                f"RL llm.rl.pt：时长 {dur:.2f}s · 用时 {elapsed:.2f}s · RTF {elapsed / max(dur, 1e-6):.3f}"
            )

        if mode == MODE_COMPARE:
            lines.append("同一参考音、同一文本，请左右试听对比音色与清晰度。")

        return rl_path, base_path, "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosyvoice-root", default=str(root / "third_party" / "CosyVoice"))
    parser.add_argument("--model-dir", default=str(root / "pretrained_models" / "Fun-CosyVoice3-0.5B"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1", help="监听地址，局域网访问可用 0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    import gradio as gr

    engine = CloneEngine(
        Path(args.cosyvoice_root).resolve(),
        Path(args.model_dir).resolve(),
        fp16=args.fp16,
        force_gpu=args.force_gpu,
    )

    with gr.Blocks(title="CosyVoice 3-0.5B 声音克隆对比") as demo:
        gr.Markdown(
            "## CosyVoice 3-0.5B 声音克隆\n"
            "上传 3–10 秒参考音频，填写逐字转写和目标文本。"
            "默认 **对比 RL 与非 RL 基座**：同一条件各合成一遍，左右试听。"
        )
        with gr.Row():
            prompt_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="参考音频")
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="参考音频转写（必填，要和录音内容一致）",
                    placeholder="例如：希望你以后能够做的比我还好呦。",
                    lines=3,
                )
                tts_text = gr.Textbox(label="要合成的文本", value=DEFAULT_TEXT, lines=4)
                instruct = gr.Textbox(
                    label="可选指令（留空=纯克隆；填写则走 instruct2，例如：请用开心的语气说）",
                    lines=2,
                )
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速")
                mode = gr.Radio(
                    choices=[MODE_COMPARE, MODE_RL, MODE_BASE],
                    value=MODE_COMPARE,
                    label="生成模式",
                )
                run_btn = gr.Button("开始生成 / 对比", variant="primary")

        gr.Markdown("### 试听对比")
        with gr.Row():
            output_rl = gr.Audio(label="RL（llm.rl.pt）", type="filepath")
            output_base = gr.Audio(label="非 RL 基座（llm.pt）", type="filepath")
        info = gr.Textbox(label="状态", interactive=False, lines=4)

        run_btn.click(
            engine.synthesize,
            inputs=[prompt_audio, prompt_text, tts_text, instruct, speed, mode],
            outputs=[output_rl, output_base, info],
        )

    demo.queue().launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(Path(__file__).resolve().parent)
    main()
