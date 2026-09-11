"""CosyVoice 3-0.5B zero-shot clone CLI (Fun-CosyVoice3-0.5B-2512 vs Qwen3-TTS).

Depends on a local FunAudioLLM/CosyVoice checkout (see README.md).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_TTS_TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
MIN_GPU_GIB_FOR_CUDA = 6.0


def wrap_prompt_text(transcript: str) -> str:
    text = transcript.strip()
    if "<|endofprompt|>" in text:
        return text
    return PROMPT_PREFIX + text


def wrap_instruct_text(instruct: str) -> str:
    """Official CosyVoice3 instruct2: style goes before <|endofprompt|>."""
    text = instruct.strip()
    if "<|endofprompt|>" in text:
        return text
    if text.startswith("You are a helpful assistant."):
        body = text[len("You are a helpful assistant.") :].strip()
        return f"You are a helpful assistant. {body}<|endofprompt|>"
    return f"You are a helpful assistant. {text}<|endofprompt|>"


def prompt_length_warning(tts_text: str, prompt_text: str) -> str | None:
    tts = tts_text.strip()
    prompt = prompt_text.strip()
    if prompt and len(tts) < 0.5 * len(prompt):
        return (
            f"目标文本过短（{len(tts)} 字）相对参考转写（{len(prompt)} 字），"
            "CosyVoice3 容易复读参考音、对不上要说的句子。请加长要合成的文本，或换更短的参考音。"
        )
    return None


def default_prompt_wav_candidates(cosyvoice_root: Path, model_dir: Optional[Path] = None) -> list:
    root = Path(__file__).resolve().parent
    paths = [
        Path(cosyvoice_root) / "asset" / "zero_shot_prompt.wav",
    ]
    if model_dir is not None:
        paths.append(Path(model_dir) / "asset" / "zero_shot_prompt.wav")
    paths.append(root / "third_party" / "CosyVoice" / "asset" / "zero_shot_prompt.wav")
    return paths


def resolve_default_prompt(cosyvoice_root: Path, model_dir: Optional[Path] = None):
    """Official CosyVoice3 demo voice: no user clone audio required."""
    for path in default_prompt_wav_candidates(cosyvoice_root, model_dir):
        if path.exists():
            return path.resolve(), DEFAULT_PROMPT_TEXT
    searched = "\n".join(f"  - {p}" for p in default_prompt_wav_candidates(cosyvoice_root, model_dir))
    raise FileNotFoundError(
        "未找到官方默认音色 zero_shot_prompt.wav。请先克隆 CosyVoice 仓库，或指定 --prompt-wav。\n" + searched
    )


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


def reset_runtime_state(cosyvoice) -> None:
    """Clear leftover decode caches between CosyVoice inferences."""
    m = getattr(cosyvoice, "model", None)
    if m is None:
        return
    for name in (
        "tts_speech_token_dict",
        "llm_end_dict",
        "mel_overlap_dict",
        "flow_cache_dict",
        "hift_cache_dict",
    ):
        cache = getattr(m, name, None)
        if isinstance(cache, dict):
            cache.clear()


def load_cosyvoice(model_dir: Path, fp16: bool, *, AutoModel=None):
    """Official AutoModel: Fun-CosyVoice3-0.5B-2512 llm.pt only."""
    if AutoModel is None:
        from cosyvoice.cli.cosyvoice import AutoModel as AutoModel

    model_dir = Path(model_dir)
    print(f"加载 Fun-CosyVoice3-0.5B-2512（官方 AutoModel / llm.pt）: {model_dir}")
    return AutoModel(model_dir=str(model_dir), load_trt=False, fp16=fp16)


def prepare_wav_tensor(speech):
    """Return CPU float32 [channel, time]. CosyVoice chunks are usually [1, T]; [T, 1] would look like 0s."""
    t = speech.detach().cpu().float().contiguous()
    if t.numel() == 0:
        raise RuntimeError("推理返回了空张量（0 个采样）。")
    t = t.squeeze()
    if t.ndim == 0:
        raise RuntimeError("推理返回了标量，没有音频。")
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim >= 3:
        t = t.reshape(1, -1)
    elif t.shape[0] > 8 and t.shape[0] > t.shape[1]:
        t = t.transpose(0, 1).contiguous()
    if t.shape[-1] < 2400:
        raise RuntimeError(
            f"合成音频过短（{t.shape[-1]} 采样，shape={tuple(t.shape)}），"
            "多半是 talker 立刻 EOS。请用官方 AutoModel 加载 llm.pt，且不要把 LLM 转成 fp16。"
        )
    return t


def concatenate_speech(chunks: list):
    import torch

    waves = []
    for chunk in chunks:
        wav = chunk.get("tts_speech") if isinstance(chunk, dict) else chunk
        if wav is None:
            continue
        waves.append(prepare_wav_tensor(wav))
    if not waves:
        raise RuntimeError(
            "推理没有返回有效音频（时长为 0）。"
            "多半是 LLM 立刻 EOS：请确认未把 talker 转成 fp16，且 prompt 含 <|endofprompt|>。"
        )
    return torch.cat(waves, dim=-1)


def run_clone(args: argparse.Namespace) -> Path:
    add_cosyvoice_to_path(Path(args.cosyvoice_root).resolve())
    maybe_force_cpu(args.force_gpu)

    import torchaudio

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "cosyvoice3.yaml").exists():
        raise FileNotFoundError(f"模型目录无效（缺少 cosyvoice3.yaml）: {model_dir}")

    if args.tts_only:
        prompt_wav, prompt_raw = resolve_default_prompt(
            Path(args.cosyvoice_root).resolve(), model_dir
        )
        print(f"纯文本 TTS：使用官方默认音色 {prompt_wav}")
    else:
        if not args.prompt_wav or not args.prompt_text:
            raise SystemExit("克隆模式需要 --prompt-wav 和 --prompt-text；纯文本 TTS 请加 --tts-only。")
        prompt_wav = Path(args.prompt_wav).resolve()
        if not prompt_wav.exists():
            raise FileNotFoundError(f"参考音频不存在: {prompt_wav}")
        prompt_raw = args.prompt_text

    print(f"加载模型: {model_dir}")
    t0 = time.time()
    cosyvoice = load_cosyvoice(model_dir, args.fp16)
    print(f"模型就绪，用时 {time.time() - t0:.1f}s，采样率 {cosyvoice.sample_rate}")

    prompt_text = wrap_prompt_text(prompt_raw)
    tts_text = args.text.strip()
    warn = prompt_length_warning(tts_text, prompt_raw)
    if warn:
        print(f"[warn] {warn}")
    print(f"prompt_text: {prompt_text}")
    print(f"tts_text   : {tts_text}")

    t1 = time.time()
    chunks = []
    if args.instruct:
        instruct = wrap_instruct_text(args.instruct)
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
    torchaudio.save(str(out_path), speech.contiguous(), cosyvoice.sample_rate)
    elapsed = time.time() - t1
    duration = speech.shape[-1] / cosyvoice.sample_rate
    print(f"已保存: {out_path}")
    print(f"音频时长 {duration:.2f}s，合成用时 {elapsed:.2f}s，RTF={elapsed / max(duration, 1e-6):.3f}")

    if args.with_qwen:
        from qwen3_tts import qwen_voice_clone

        qwen_name = Path(args.out_name).stem + "_qwen3.wav"
        qwen_path = out_dir / qwen_name
        tq = time.time()
        qwen_wav, qwen_sr = qwen_voice_clone(tts_text, str(prompt_wav), prompt_raw)
        torchaudio.save(str(qwen_path), qwen_wav.contiguous(), qwen_sr)
        q_elapsed = time.time() - tq
        q_dur = qwen_wav.shape[-1] / qwen_sr
        print(f"已保存 Qwen3-TTS: {qwen_path}")
        print(
            f"Qwen3 时长 {q_dur:.2f}s，合成用时 {q_elapsed:.2f}s，RTF={q_elapsed / max(q_dur, 1e-6):.3f}"
        )

    return out_path


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fun-CosyVoice3-0.5B-2512 声音克隆 demo")
    parser.add_argument("--cosyvoice-root", default=str(root / "third_party" / "CosyVoice"))
    parser.add_argument(
        "--model-dir",
        default=str(root / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
    )
    parser.add_argument("--prompt-wav", default="", help="参考音频；--tts-only 时可不填")
    parser.add_argument("--prompt-text", default="", help="参考音频转写；--tts-only 时可不填")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--instruct", default="", help="可选：风格指令，走 inference_instruct2")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(root / "outputs"))
    parser.add_argument("--out-name", default="clone.wav")
    parser.add_argument("--tts-only", action="store_true", help="纯文本 TTS，使用官方 zero_shot_prompt 默认音色")
    parser.add_argument("--fp16", action="store_true", help="仅在显存足够的 GPU 上建议开启")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--with-qwen",
        action="store_true",
        help="额外用 Qwen3-TTS-12Hz-0.6B-Base 克隆同一参考音",
    )
    return parser


if __name__ == "__main__":
    run_clone(build_parser().parse_args())
