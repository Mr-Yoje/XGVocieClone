"""Gradio UI for CosyVoice 3-0.5B-RL voice cloning, with base vs RL compare."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

from demo_clone import (
    DEFAULT_TTS_TEXT,
    add_cosyvoice_to_path,
    apply_llm_weights,
    concatenate_speech,
    llm_checkpoint_path,
    load_independent_cosyvoices,
    maybe_force_cpu,
    prompt_length_warning,
    reset_runtime_state,
    resolve_default_prompt,
    wrap_instruct_text,
    wrap_prompt_text,
)


DEFAULT_TEXT = "你好，这是一次 Fun-CosyVoice 3 零样本声音克隆测试。希望你能听出和参考音频相近的音色。"
MODE_TTS = "纯文本 TTS（默认音色）"
MODE_COMPARE = "克隆：对比 RL / 基座 / Qwen3"
MODE_RL = "克隆：仅 RL"
MODE_BASE = "克隆：仅基座"


def _quiet_http_loggers() -> None:
    """httpx/httpcore 默认会把 TLS 握手打成 DEBUG，淹没监听地址。"""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    logging.basicConfig(level=logging.INFO, force=False)
    logging.getLogger().setLevel(logging.INFO)
    for name in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.proxy",
        "urllib3",
        "urllib3.connectionpool",
        "huggingface_hub",
        "huggingface_hub.utils",
        "asyncio",
        "gradio",
        "gradio_client",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


class CloneEngine:
    def __init__(
        self,
        cosyvoice_root: Path,
        model_dir: Path,
        fp16: bool,
        force_gpu: bool,
        shared_talker: bool = False,
    ):
        add_cosyvoice_to_path(cosyvoice_root)
        maybe_force_cpu(force_gpu)
        if not (model_dir / "cosyvoice3.yaml").exists():
            raise FileNotFoundError(f"模型目录无效: {model_dir}")
        self.cosyvoice_root = cosyvoice_root
        self.model_dir = model_dir
        self.fp16 = fp16
        self.want_shared = shared_talker
        self.shared_talker = shared_talker
        self.base_ckpt = llm_checkpoint_path(model_dir, "base")
        self.rl_ckpt = llm_checkpoint_path(model_dir, "rl")
        self.models = None
        self.active = "base"
        print("先启动网页；CosyVoice 权重在首次点「生成」时再加载。", flush=True)

    def _ensure_loaded(self) -> None:
        if self.models is not None:
            return
        from cosyvoice.cli.cosyvoice import AutoModel

        print(f"加载模型: {self.model_dir}", flush=True)
        t0 = time.time()
        self.models, self.shared_talker = load_independent_cosyvoices(
            self.model_dir,
            self.fp16,
            shared_talker=self.want_shared,
            AutoModel=AutoModel,
            rl_ckpt=self.rl_ckpt,
        )
        self.active = "base"
        print(f"模型就绪 {time.time() - t0:.1f}s（独立实例={not self.shared_talker}）", flush=True)

    def _prepare_talker(self, variant: str):
        model = self.models[variant]
        if self.shared_talker:
            if variant != self.active:
                path = self.rl_ckpt if variant == "rl" else self.base_ckpt
                apply_llm_weights(model, path, label="RL" if variant == "rl" else "基座")
                self.active = variant
            else:
                reset_runtime_state(model)
            return model
        reset_runtime_state(model)
        return model

    def _infer(self, variant: str, prompt_wav: str, prompt_text: str, tts_text: str, instruct: str, speed: float):
        model = self._prepare_talker(variant)
        chunks = []
        if instruct and instruct.strip():
            for item in model.inference_instruct2(
                tts_text.strip(),
                wrap_instruct_text(instruct.strip()),
                prompt_wav,
                stream=False,
                speed=float(speed),
            ):
                chunks.append(item)
        else:
            for item in model.inference_zero_shot(
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

    def _save(self, speech, tag: str, sample_rate: int | None = None) -> tuple[str, float]:
        import torchaudio

        sr = int(sample_rate or self.models["base"].sample_rate)
        dur = speech.shape[-1] / sr
        stamp = int(time.time() * 1000)
        tmp = Path(tempfile.gettempdir()) / f"cosyvoice3_{tag}_{stamp}.wav"
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        persistent = out_dir / f"{tag}_{stamp}.wav"
        torchaudio.save(str(tmp), speech.contiguous(), sr)
        torchaudio.save(str(persistent), speech.contiguous(), sr)
        print(f"保存 {tag}: shape={tuple(speech.shape)} dur={dur:.3f}s path={persistent}")
        return str(tmp), dur

    def _infer_qwen(self, prompt_wav: str, prompt_text: str, tts_text: str):
        from qwen3_tts import qwen_voice_clone

        return qwen_voice_clone(tts_text, prompt_wav, prompt_text)

    def synthesize(self, prompt_audio, prompt_text: str, tts_text: str, instruct: str, speed: float, mode: str, with_qwen: bool):
        if not tts_text or not tts_text.strip():
            raise ValueError("请填写要合成的文本。")
        self._ensure_loaded()

        tts_only = mode == MODE_TTS
        if tts_only:
            prompt_wav, prompt_raw = resolve_default_prompt(self.cosyvoice_root, self.model_dir)
            prompt_text = prompt_raw
        else:
            if prompt_audio is None:
                raise ValueError("克隆模式请先上传或录制参考音频。")
            if not prompt_text or not prompt_text.strip():
                raise ValueError("克隆模式请填写参考音频的逐字转写。")
            prompt_wav = prompt_audio
            prompt_raw = prompt_text

        want_rl = mode in (MODE_COMPARE, MODE_RL, MODE_TTS)
        want_base = mode in (MODE_COMPARE, MODE_BASE, MODE_TTS)
        lines = []
        rl_path = None
        base_path = None
        qwen_path = None

        if tts_only:
            lines.append(f"纯文本 TTS，默认音色：{prompt_wav}")
        warn = prompt_length_warning(tts_text, prompt_raw)
        if warn:
            lines.append(warn)

        if self.shared_talker:
            lines.append("显存不足或指定了 --shared-talker：基座与 RL 共用引擎并切换权重。")
        else:
            lines.append("基座与 RL 为两套独立 CosyVoice（同一输入，各自推理、各自输出）。")

        if want_base:
            t0 = time.time()
            speech = self._infer("base", prompt_wav, prompt_raw, tts_text, instruct, speed)
            base_path, dur = self._save(speech, "base")
            elapsed = time.time() - t0
            lines.append(
                f"CosyVoice 基座：时长 {dur:.2f}s · 用时 {elapsed:.2f}s · RTF {elapsed / max(dur, 1e-6):.3f}"
            )

        if want_rl:
            t0 = time.time()
            speech = self._infer("rl", prompt_wav, prompt_raw, tts_text, instruct, speed)
            rl_path, dur = self._save(speech, "rl")
            elapsed = time.time() - t0
            lines.append(
                f"CosyVoice RL：时长 {dur:.2f}s · 用时 {elapsed:.2f}s · RTF {elapsed / max(dur, 1e-6):.3f}"
            )

        if with_qwen:
            try:
                t0 = time.time()
                speech, sr = self._infer_qwen(str(prompt_wav), prompt_raw, tts_text)
                qwen_path, dur = self._save(speech, "qwen3_0.6b", sample_rate=sr)
                elapsed = time.time() - t0
                lines.append(
                    f"Qwen3-TTS-0.6B-Base：时长 {dur:.2f}s · 用时 {elapsed:.2f}s · RTF {elapsed / max(dur, 1e-6):.3f}"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"Qwen3-TTS-0.6B 失败：{exc}")

        if mode == MODE_COMPARE:
            lines.append("同一参考音、同一文本，请试听对比音色与清晰度。")
        if tts_only:
            lines.append("未使用上传的参考音；CosyVoice 用官方默认音色，Qwen3-TTS 克隆同一段默认参考音。")

        return rl_path, base_path, qwen_path, "\n".join(lines)


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
    parser.add_argument("--no-qwen", action="store_true", help="不加载、不对比 Qwen3-TTS-0.6B")
    parser.add_argument(
        "--shared-talker",
        action="store_true",
        help="基座与 RL 共用一份 CosyVoice（省显存，会切换权重）",
    )
    args = parser.parse_args()
    _quiet_http_loggers()

    import gradio as gr

    engine = CloneEngine(
        Path(args.cosyvoice_root).resolve(),
        Path(args.model_dir).resolve(),
        fp16=args.fp16,
        force_gpu=args.force_gpu,
        shared_talker=args.shared_talker,
    )
    _quiet_http_loggers()

    with gr.Blocks(title="CosyVoice 3 / Qwen3-TTS 声音对比") as demo:
        gr.Markdown(
            "## CosyVoice 3-0.5B 与 Qwen3-TTS-0.6B\n"
            "- **纯文本 TTS**：不需要参考音；CosyVoice 用官方默认音色，Qwen3-TTS 克隆同一段默认参考音。\n"
            "- **克隆**：上传 3–10 秒参考音频并填写转写，再输入要说的新文本。\n"
            "- **基座 / RL**：同一参考音和同一句文本，两套独立 CosyVoice 分别推理，两路独立输出。\n"
            "- 页面会马上打开；**第一次点生成**才会加载模型（可能要几分钟）。"
        )
        with gr.Row():
            prompt_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="参考音频")
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="参考音频转写（必填，要和录音内容一致）",
                    placeholder="例如：希望你以后能够做的比我还好呦。",
                    lines=3,
                )
                tts_text = gr.Textbox(label="要合成的文本", value=DEFAULT_TTS_TEXT, lines=4)
                instruct = gr.Textbox(
                    label="可选指令（留空=普通合成；填写则走 instruct2，例如：请用开心的语气说）",
                    lines=2,
                )
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速")
                mode = gr.Radio(
                    choices=[MODE_TTS, MODE_COMPARE, MODE_RL, MODE_BASE],
                    value=MODE_TTS,
                    label="生成模式",
                )
                with_qwen = gr.Checkbox(
                    value=not args.no_qwen,
                    label="同时对比 Qwen3-TTS-12Hz-0.6B-Base（首次会下载权重，显存紧张可关掉）",
                )
                run_btn = gr.Button("开始生成 / 对比", variant="primary")

        gr.Markdown("### 试听对比")
        with gr.Row():
            output_rl = gr.Audio(label="CosyVoice RL（llm.rl.pt）", type="filepath")
            output_base = gr.Audio(label="CosyVoice 基座（llm.pt）", type="filepath")
            output_qwen = gr.Audio(label="Qwen3-TTS-0.6B-Base", type="filepath")
        info = gr.Textbox(label="状态", interactive=False, lines=6)

        run_btn.click(
            engine.synthesize,
            inputs=[prompt_audio, prompt_text, tts_text, instruct, speed, mode, with_qwen],
            outputs=[output_rl, output_base, output_qwen, info],
        )

    def _on_signal(signum, _frame):
        print(f"\n收到信号 {signum}，正在关闭 Gradio…", flush=True)
        try:
            demo.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    print("=" * 64, flush=True)
    print(f"监听地址: http://{args.server_name}:{args.port}", flush=True)
    print(f"本机打开: http://127.0.0.1:{args.port}", flush=True)
    if args.share:
        print("正在申请 Gradio 公网链接（*.gradio.live），请等十几秒…", flush=True)
        print("若一直没有 live 链接：Colab 请跑「端口转发」格。", flush=True)
    print("=" * 64, flush=True)

    import threading

    def _colab_port_hint() -> None:
        time.sleep(2)
        try:
            from google.colab import output  # type: ignore

            print(
                f"Colab 端口转发（不依赖 gradio.live）: 见下方窗口，或新单元格运行\n"
                f"  from google.colab import output\n"
                f"  output.serve_kernel_port_as_window({args.port})",
                flush=True,
            )
            output.serve_kernel_port_as_window(args.port)
        except Exception:
            pass

    threading.Thread(target=_colab_port_hint, daemon=True).start()

    try:
        demo.queue().launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False,
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt，正在关闭…", flush=True)
        try:
            demo.close()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(Path(__file__).resolve().parent)
    main()
