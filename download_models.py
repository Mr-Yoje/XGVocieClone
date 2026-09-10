"""Download Fun-CosyVoice3-0.5B (contains llm.pt + llm.rl.pt)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Fun-CosyVoice3-0.5B-2512 权重")
    parser.add_argument(
        "--out",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="本地模型目录",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="下载源：国内优先 modelscope",
    )
    parser.add_argument("--with-ttsfrd", action="store_true", help="同时下载 ttsfrd 文本正规化资源（可选）")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = args.source
    if source == "auto":
        try:
            import modelscope  # noqa: F401

            source = "modelscope"
        except ImportError:
            source = "huggingface"

    print(f"使用 {source} 下载到 {out}")
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=str(out))
        if args.with_ttsfrd:
            ttsfrd = out.parent / "CosyVoice-ttsfrd"
            snapshot_download("iic/CosyVoice-ttsfrd", local_dir=str(ttsfrd))
    else:
        from huggingface_hub import snapshot_download

        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=str(out))
        if args.with_ttsfrd:
            ttsfrd = out.parent / "CosyVoice-ttsfrd"
            snapshot_download("FunAudioLLM/CosyVoice-ttsfrd", local_dir=str(ttsfrd))

    rl = out / "llm.rl.pt"
    base = out / "llm.pt"
    print(f"llm.pt     : {'OK' if base.exists() else 'MISSING'}  {base}")
    print(f"llm.rl.pt  : {'OK' if rl.exists() else 'MISSING'}  {rl}")
    print("下载完成。")


if __name__ == "__main__":
    main()
