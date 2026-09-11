"""CosyVoice 3-0.5B-RL zero-shot voice clone CLI.

Depends on a local FunAudioLLM/CosyVoice checkout (see README.md).
"""

from __future__ import annotations

import argparse
import os
import shutil
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


def llm_checkpoint_path(model_dir: Path, variant: str) -> Path:
    name = "llm.rl.pt" if variant == "rl" else "llm.pt"
    path = model_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {variant} 权重 {path}。请确认已下载 Fun-CosyVoice3-0.5B-2512。"
        )
    return path


def coerce_llm_state_dict(state):
    """Use raw nn.Module state_dict as-is. Only unwrap training checkpoint wrappers."""
    if not isinstance(state, dict) or not state:
        return state
    weight_like = 0
    for value in state.values():
        if isinstance(value, dict):
            continue
        if isinstance(value, (int, float, str, bool, type(None), list, tuple)):
            continue
        weight_like += 1
    if weight_like >= max(1, (len(state) + 1) // 2):
        first = next(iter(state))
        if isinstance(first, str) and first.startswith("module."):
            return {k[len("module.") :]: v for k, v in state.items()}
        return state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return coerce_llm_state_dict(state["state_dict"])
    if "model" in state and isinstance(state["model"], dict):
        return coerce_llm_state_dict(state["model"])
    first = next(iter(state))
    if isinstance(first, str) and first.startswith("module."):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def reset_runtime_state(cosyvoice) -> None:
    """Clear leftover decode caches so swapping llm.pt / llm.rl.pt does not yield empty audio."""
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


def apply_llm_weights(cosyvoice, ckpt_path: Path, label: str = "") -> None:
    """Load talker the same way CosyVoiceModel.load does: fp32 state_dict, no .half()."""
    import torch

    device = getattr(cosyvoice.model, "device", None)
    if device is None:
        device = next(cosyvoice.model.llm.parameters()).device
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    state = coerce_llm_state_dict(raw)
    result = cosyvoice.model.llm.load_state_dict(state, strict=False)
    if isinstance(result, tuple):
        missing, unexpected = list(result[0]), list(result[1])
    else:
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
    tag = label or ckpt_path.name
    if missing:
        raise RuntimeError(f"{tag} 权重缺少 {len(missing)} 个参数，例如 {missing[:5]}")
    if unexpected:
        print(f"[warn] {tag} 忽略多余键 {len(unexpected)} 个，例如 {unexpected[:8]}")
    # Official path keeps fp32 weights and uses autocast when fp16=True. Casting the
    # whole LLM to half here makes RL emit 0s and the base talker babble.
    cosyvoice.model.llm.to(device).eval()
    reset_runtime_state(cosyvoice)
    print(f"已加载 talker 权重 ({tag}): {ckpt_path}")


def apply_rl_weights(cosyvoice, model_dir: Path) -> None:
    apply_llm_weights(cosyvoice, llm_checkpoint_path(model_dir, "rl"), label="RL")


def _is_oom(exc: BaseException) -> bool:
    if "OutOfMemory" in type(exc).__name__:
        return True
    return "out of memory" in str(exc).lower()


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def prepare_official_llm_dir(model_dir: Path, llm_ckpt: Path) -> Path:
    """Directory whose llm.pt is the given checkpoint, so AutoModel.load uses official strict=True.

    Overlaying llm.rl.pt onto an already-constructed llm.pt instance often makes RL EOS immediately.
    """
    model_dir = Path(model_dir).resolve()
    llm_ckpt = Path(llm_ckpt).resolve()
    if llm_ckpt == model_dir / "llm.pt":
        return model_dir
    shadow = model_dir.parent / f".cosyvoice_llm_{llm_ckpt.stem}"
    shadow.mkdir(parents=True, exist_ok=True)
    for item in model_dir.iterdir():
        if item.name == "llm.pt":
            continue
        _link_or_copy(item, shadow / item.name)
    dest_llm = shadow / "llm.pt"
    if dest_llm.exists() or dest_llm.is_symlink():
        dest_llm.unlink()
    _link_or_copy(llm_ckpt, dest_llm)
    return shadow


def load_independent_cosyvoices(
    model_dir: Path,
    fp16: bool,
    *,
    shared_talker: bool = False,
    AutoModel=None,
    apply_rl=None,
    rl_ckpt: Optional[Path] = None,
):
    """Load base and RL as two AutoModel instances. Same input later, separate weights/outputs.

    If VRAM cannot hold two copies, fall back to one shared engine (caller must swap weights).
    """
    if AutoModel is None:
        from cosyvoice.cli.cosyvoice import AutoModel as AutoModel

    apply_rl = apply_rl or apply_llm_weights
    model_dir = Path(model_dir)
    rl_ckpt = Path(rl_ckpt) if rl_ckpt is not None else llm_checkpoint_path(model_dir, "rl")
    print("加载 CosyVoice 基座（llm.pt）")
    base = AutoModel(model_dir=str(model_dir), load_trt=False, fp16=fp16)
    if shared_talker:
        print("shared_talker：基座与 RL 共用一份引擎，推理时切换权重")
        return {"base": base, "rl": base}, True
    rl_dir = prepare_official_llm_dir(model_dir, rl_ckpt)
    print(f"再加载一份 CosyVoice，官方路径读取 RL：{rl_dir / 'llm.pt'} -> {rl_ckpt}")
    try:
        rl = AutoModel(model_dir=str(rl_dir), load_trt=False, fp16=fp16)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        print(f"[warn] 第二份 CosyVoice 显存不足，回退共用引擎: {exc}")
        apply_rl(base, rl_ckpt, label="RL")
        return {"base": base, "rl": base}, True
    if rl is base:
        raise RuntimeError("RL 与基座必须是两个独立实例")
    print("基座与 RL 已独立加载（各自走官方 llm.pt 加载），之后不再互相覆盖权重")
    return {"base": base, "rl": rl}, False


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
            "多半是 talker 立刻 EOS。RL 必须用官方方式加载 llm.rl.pt，且不要把 LLM 转成 fp16。"
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
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "cosyvoice3.yaml").exists():
        raise FileNotFoundError(f"模型目录无效（缺少 cosyvoice3.yaml）: {model_dir}")

    if getattr(args, "tts_only", False):
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
    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, fp16=args.fp16)
    if args.use_rl:
        apply_rl_weights(cosyvoice, model_dir)
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

    if getattr(args, "with_qwen", False):
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
    parser = argparse.ArgumentParser(description="Fun-CosyVoice3-0.5B-RL 声音克隆 demo")
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
    parser.add_argument("--use-rl", dest="use_rl", action="store_true", default=True)
    parser.add_argument("--use-base", dest="use_rl", action="store_false", help="改用 llm.pt（非 RL）")
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
