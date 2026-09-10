# Setup CosyVoice runtime under this project (Windows).
# Requires: git, conda (Miniconda / Anaconda), NVIDIA driver if using GPU.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$CosyDir = Join-Path $Root "third_party\CosyVoice"
if (-not (Test-Path $CosyDir)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "third_party") | Out-Null
    Write-Host "Cloning FunAudioLLM/CosyVoice (with submodules)..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $CosyDir
} else {
    Write-Host "CosyVoice already exists: $CosyDir"
    Push-Location $CosyDir
    git submodule update --init --recursive
    Pop-Location
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-Host "未检测到 conda。请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    Write-Host "然后重新打开终端，再运行本脚本。"
    exit 1
}

conda create -n cosyvoice -y python=3.10
conda run -n cosyvoice python -m pip install -U pip
conda run -n cosyvoice python -m pip install -r (Join-Path $CosyDir "requirements.txt") -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
conda run -n cosyvoice python -m pip install modelscope huggingface_hub gradio torchaudio

Write-Host ""
Write-Host "环境已创建。下一步："
Write-Host "  conda activate cosyvoice"
Write-Host "  python download_models.py"
Write-Host "  python webui_clone.py"
Write-Host ""
Write-Host "注意：本机若只有 2GB 显存（如 MX450），脚本会自动走 CPU，首次推理会较慢。"
