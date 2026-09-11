"""Qwen3-TTS-12Hz-0.6B-Base voice clone helper (same ref audio as CosyVoice)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
_MODEL = None
_LOAD_ERROR: Optional[str] = None


def detect_language(text: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "Chinese"
    return "English"


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _pick_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _patch_check_model_inputs() -> None:
    """qwen-tts uses @check_model_inputs(); transformers 4.57 expects @check_model_inputs."""
    try:
        import transformers.utils.generic as generic
    except Exception:
        return
    orig = getattr(generic, "check_model_inputs", None)
    if orig is None or getattr(orig, "_xg_compat", False):
        return

    def check_model_inputs(func=None, *args, **kwargs):
        if func is None:

            def decorator(inner):
                try:
                    return orig(inner, *args, **kwargs)
                except TypeError:
                    wrapped = orig(*args, **kwargs)
                    return wrapped(inner) if callable(wrapped) else inner

            return decorator
        return orig(func, *args, **kwargs)

    check_model_inputs._xg_compat = True  # type: ignore[attr-defined]
    generic.check_model_inputs = check_model_inputs
    try:
        import transformers.utils as utils

        if getattr(utils, "check_model_inputs", None) is orig:
            utils.check_model_inputs = check_model_inputs
    except Exception:
        pass


def get_qwen_model(model_id: str = QWEN_MODEL_ID):
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR:
        raise RuntimeError(_LOAD_ERROR)
    _patch_check_model_inputs()
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        _LOAD_ERROR = "未安装 qwen-tts。请执行: pip install -U qwen-tts"
        raise RuntimeError(_LOAD_ERROR) from exc

    device = _pick_device()
    if device == "cpu":
        print("[qwen3-tts] 使用 CPU（可能很慢）")
    dtype = _pick_dtype()
    local = Path("pretrained_models") / "Qwen3-TTS-12Hz-0.6B-Base"
    source = str(local.resolve()) if (local / "config.json").exists() else model_id
    print(f"[qwen3-tts] 加载 {source} device={device} dtype={dtype}")

    last_err = None
    for attn in ("sdpa", "eager", "flash_attention_2"):
        try:
            kwargs = dict(device_map=device, dtype=dtype)
            if attn:
                kwargs["attn_implementation"] = attn
            _MODEL = Qwen3TTSModel.from_pretrained(source, **kwargs)
            print(f"[qwen3-tts] attn={attn} 就绪")
            return _MODEL
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[qwen3-tts] attn={attn} 失败: {exc}")
    _LOAD_ERROR = f"Qwen3-TTS 加载失败: {last_err}"
    raise RuntimeError(_LOAD_ERROR) from last_err


def qwen_voice_clone(text: str, ref_audio: str, ref_text: str):
    """Return (waveform [1, T] CPU float32, sample_rate)."""
    import numpy as np
    import torch

    _patch_check_model_inputs()
    model = get_qwen_model()
    wavs, sr = model.generate_voice_clone(
        text=text.strip(),
        language=detect_language(text),
        ref_audio=ref_audio,
        ref_text=ref_text.strip(),
        x_vector_only_mode=False,
    )
    wav = wavs[0]
    if isinstance(wav, np.ndarray):
        t = torch.from_numpy(np.asarray(wav)).float()
    else:
        t = wav.detach().cpu().float()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(1, -1)
    if t.shape[0] > t.shape[-1] and t.shape[0] > 8:
        t = t.transpose(0, 1).contiguous()
    return t.contiguous(), int(sr)
