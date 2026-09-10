"""CosyVoice 3-0.5B-RL zero-shot voice clone CLI.

Depends on a local FunAudioLLM/CosyVoice checkout (see README.md).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
MIN_GPU_GIB_FOR_CUDA = 6.0


def wrap_prompt_text(transcript: str) -> str:
    text = transcript.strip()
    if "<|endofprompt|>" in text:
        return text
    return PROMPT_PREFIX + text


def maybe_force_cpu(force_gpu: bool) -> None:
    """MX450-class 2GB cards cannot hold CosyVoice3; hide CUDA unless forced."""
    if force_gpu:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gib < MIN_GPU_GIB_FOR_CUDA:
            print(
                f"[warn] GPU 显存约 {vram_gib:.1f} GB，低于建议 {MIN_GPU_GIB_FOR_CUDA:.0f} GB，"
                "改用 CPU。如需强制走 GPU，请加 --force-gpu"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # torch already imported: subsequent CUDA checks may still see the device.
            # CosyVoice reads torch.cuda.is_available(); hide it if possible.
            torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 检测 GPU 失败，继续: {exc}")


def add_cosyvoice_to_path(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"未找到 CosyVoice 仓库: {root}\n请先运行 setup_env.ps1，或用 --cosyvoice-root 指定路径。"
        )
    sys.path.insert(0, str(root))
    matcha = root / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))
    else:
        print(f"[warn] 未找到 {matcha}，若 import 失败请 git submodule update --init --recursive")


def llm_checkpoint_path(model_dir: Path, variant: str) -> Path:
    name = "llm.rl.pt" if variant == "rl" else "llm.pt"
    path = model_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {variant} 权重 {path}。请确认已下载 Fun-CosyVoice3-0.5B-2512。"
        )
    return path


def apply_llm_weights(cosyvoice, ckpt_path: Path, label: str = "") -> None:
    import torch

    device = getattr(cosyvoice.model, "device", None)
    if device is None:
        device = next(cosyvoice.model.llm.parameters()).device
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    cosyvoice.model.llm.load_state_dict(state, strict=True)
    cosyvoice.model.llm.to(device).eval()
    tag = label or ckpt_path.name
    print(f"已加载 talker 权重 ({tag}): {ckpt_path}")


def apply_rl_weights(cosyvoice, model_dir: Path) -> None:
    apply_llm_weights(cosyvoice, llm_checkpoint_path(model_dir, "rl"), label="RL")


def concatenate_speech(chunks: list) -> "object":
    import torch

    return torch.cat([c["tts_speech"] for c in chunks], dim=1)


def run_clone(args: argparse.Namespace) -> Path:
    add_cosyvoice_to_path(Path(args.cosyvoice_root).resolve())
    maybe_force_cpu(args.force_gpu)

    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "cosyvoice3.yaml").exists():
        raise FileNotFoundError(f"模型目录无效（缺少 cosyvoice3.yaml）: {model_dir}")

    prompt_wav = Path(args.prompt_wav).resolve()
    if not prompt_wav.exists():
        raise FileNotFoundError(f"参考音频不存在: {prompt_wav}")

    print(f"加载模型: {model_dir}")
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, fp16=args.fp16)
    if args.use_rl:
        apply_rl_weights(cosyvoice, model_dir)
    print(f"模型就绪，用时 {time.time() - t0:.1f}s，采样率 {cosyvoice.sample_rate}")

    prompt_text = wrap_prompt_text(args.prompt_text)
    tts_text = args.text.strip()
    print(f"prompt_text: {prompt_text}")
    print(f"tts_text   : {tts_text}")

    t1 = time.time()
    chunks = []
    if args.instruct:
        instruct = wrap_prompt_text(args.instruct)
        for item in cosyvoice.inference_instruct2(
            tts_text,
            instruct,
            str(prompt_wav),
            stream=False,
            speed=args.speed,
        ):
            chunks.append(item)
    else:
        for item in cosyvoice.inference_zero_shot(
            tts_text,
            prompt_text,
            str(prompt_wav),
            stream=False,
            speed=args.speed,
        ):
            chunks.append(item)

    if not chunks:
        raise RuntimeError("推理没有返回音频。")

    speech = concatenate_speech(chunks)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    torchaudio.save(str(out_path), speech, cosyvoice.sample_rate)
    elapsed = time.time() - t1
    duration = speech.shape[1] / cosyvoice.sample_rate
    print(f"已保存: {out_path}")
    print(f"音频时长 {duration:.2f}s，合成用时 {elapsed:.2f}s，RTF={elapsed / max(duration, 1e-6):.3f}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fun-CosyVoice3-0.5B-RL 声音克隆 demo")
    parser.add_argument("--cosyvoice-root", default=str(root / "third_party" / "CosyVoice"))
    parser.add_argument(
        "--model-dir",
        default=str(root / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
    )
    parser.add_argument("--prompt-wav", required=True, help="参考音频，建议 3–10 秒、尽量干净")
    parser.add_argument("--prompt-text", required=True, help="参考音频的逐字转写（说了什么）")
    parser.add_argument("--text", required=True, help="要用克隆音色说的目标文本")
    parser.add_argument("--instruct", default="", help="可选：风格指令，走 inference_instruct2")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(root / "outputs"))
    parser.add_argument("--out-name", default="clone.wav")
    parser.add_argument("--use-rl", dest="use_rl", action="store_true", default=True)
    parser.add_argument("--use-base", dest="use_rl", action="store_false", help="改用 llm.pt（非 RL）")
    parser.add_argument("--fp16", action="store_true", help="仅在显存足够的 GPU 上建议开启")
    parser.add_argument("--force-gpu", action="store_true")
    return parser


if __name__ == "__main__":
    run_clone(build_parser().parse_args())
