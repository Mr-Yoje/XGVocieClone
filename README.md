# CosyVoice 3-0.5B-RL 声音克隆 Demo

用官方 **Fun-CosyVoice3-0.5B-2512** 做零样本克隆。仓库里同时有：

| 文件 | 含义 |
| --- | --- |
| `llm.pt` | 基座 talker |
| `llm.rl.pt` | RL 后的 talker（CER/WER 更低，本 demo 默认用这个） |

官方 `AutoModel` 只会加载 `llm.pt`。本仓库在加载后把 `llm.rl.pt` 灌进 LLM，即 **3-0.5B-RL**。

## 环境要求

- Git、Miniconda（Python 3.10）；或 **Google Colab T4 GPU**
- 建议 GPU 显存 **≥ 6 GB**。2 GB 级显卡（例如 MX450）会自动改走 CPU，能跑但很慢。
- 参考音频：**3–10 秒**、单人、尽量少噪声；**必须提供逐字转写**。

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr-Yoje/XGVocieClone/blob/main/colab_clone.ipynb)

1. 打开上面的徽章（或手动打开 [`colab_clone.ipynb`](https://colab.research.google.com/github/Mr-Yoje/XGVocieClone/blob/main/colab_clone.ipynb)）。
2. **代码执行程序 → 更改运行时类型 → T4 GPU**。
3. 依次运行单元格。启动格会**后台**拉起 WebUI；日志里的 Gradio 公网链接即可试用。要停止请运行笔记本里的 **「停止 / 退出 WebUI」**（`bash colab_webui.sh stop`），不要只点中断。

Colab 使用预装的 CUDA PyTorch，不走 conda。脚本见 `setup_colab.sh`。

## 一键准备（Windows PowerShell）

```powershell
cd D:\Document\PAProject\XGVocieClone
.\setup_env.ps1
conda activate cosyvoice
python download_models.py
```

## 一键启动（Ubuntu）

```bash
cd /path/to/XGVocieClone
chmod +x setup_env.sh start_ubuntu.sh
bash start_ubuntu.sh
```

脚本会：安装 git/sox/ffmpeg（需 sudo）→ 没有 conda 则装 Miniconda 到项目内 `.miniconda3` → 克隆 CosyVoice → 建 `cosyvoice` 环境 → 下载权重 → 启动 Web 界面。

常用参数：

```bash
bash start_ubuntu.sh --listen          # 监听 0.0.0.0，局域网可访问
bash start_ubuntu.sh --port 7860
bash start_ubuntu.sh --fp16 --force-gpu
bash start_ubuntu.sh --setup-only      # 只装环境
bash start_ubuntu.sh --skip-download   # 权重已下好时跳过下载
```

只装环境、自己启动也可以：

```bash
bash setup_env.sh
source .miniconda3/etc/profile.d/conda.sh   # 或你本机的 conda.sh
conda activate cosyvoice
python download_models.py
python webui_clone.py
```

国内网络优先走 ModelScope；若失败：

```powershell
python download_models.py --source huggingface
```

权重约 **7 GB+**（含 `llm.pt`、`llm.rl.pt`、flow、speech tokenizer）。

## 命令行克隆

把参考音频放到 `prompts\`，例如 `prompts\ref.wav`。

```powershell
conda activate cosyvoice
python demo_clone.py `
  --prompt-wav prompts\ref.wav `
  --prompt-text "希望你以后能够做的比我还好呦。" `
  --text "你好，这是一次 Fun-CosyVoice 3 零样本声音克隆测试。"
```

输出：`outputs\clone.wav`。

对比基座（非 RL）：

```powershell
python demo_clone.py --use-base --prompt-wav prompts\ref.wav --prompt-text "……" --text "……" --out-name clone_base.wav
```

带情感/方言指令（`inference_instruct2`）：

```powershell
python demo_clone.py `
  --prompt-wav prompts\ref.wav `
  --prompt-text "希望你以后能够做的比我还好呦。" `
  --text "今天天气真好，我们出去走走吧。" `
  --instruct "请用开心的语气说"
```

若已手动克隆官方仓库，加上：

```text
--cosyvoice-root 路径\CosyVoice
```

官方自带参考音（克隆 CosyVoice 仓库后）：

```powershell
python demo_clone.py `
  --prompt-wav third_party\CosyVoice\asset\zero_shot_prompt.wav `
  --prompt-text "希望你以后能够做的比我还好呦。" `
  --text "八百标兵奔北坡，北坡炮兵并排跑。"
```

## Web 界面

```powershell
conda activate cosyvoice
python webui_clone.py
```

浏览器打开 `http://127.0.0.1:7860`。默认模式是 **纯文本 TTS（官方默认音色）**：只填要说的句子，对比 CosyVoice RL、基座，以及 **Qwen3-TTS-12Hz-0.6B-Base**（同一默认参考音）。克隆请改选「克隆：对比 RL / 基座 / Qwen3」并上传 3–10 秒参考音。基座与 RL 默认各加载一份 CosyVoice（同一输入、两路独立输出）；T4 显存不够可加 `--shared-talker` 或关掉 Qwen（`--no-qwen`）。

首次对比 Qwen 会从 Hugging Face 拉取 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`（也可预先放到 `pretrained_models/Qwen3-TTS-12Hz-0.6B-Base`）。需安装：`pip install -U qwen-tts`。

命令行同时出 Qwen 结果：

```powershell
python demo_clone.py --tts-only --with-qwen --text "八百标兵奔北坡，北坡炮兵并排跑。"
```

命令行纯文本 TTS：

```powershell
python demo_clone.py --tts-only --text "八百标兵奔北坡，北坡炮兵并排跑。"
```

## 使用注意

1. CosyVoice 3 的 prompt 会自动加上 `You are a helpful assistant.<|endofprompt|>`，你只需填录音原文。
2. 目标文本过短、明显短于参考转写时，官方会告警，效果可能变差。
3. 首次 CPU 推理可能要数分钟；有 8GB+ 显存时可加 `--fp16 --force-gpu`。`--fp16` 只启用官方 autocast，不会把 `llm.pt` / `llm.rl.pt` 整模转成 half（转 half 会导致 RL 0 秒、基座乱说）。
4. 仅用于你有权使用的声音样本（本人或已授权音色）。
