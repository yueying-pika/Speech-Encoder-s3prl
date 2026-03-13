# Speech-Encoder-s3prl (Qwen-Omni Upstream)

本项目基于 [s3prl](https://github.com/s3prl/s3prl) 框架，实现了 **Qwen-Omni** 系列模型的语音编码器（Speech Encoder）集成。通过该项目，用户可以方便地提取 Qwen-Omni 的上游特征，并将其应用于语音识别（ASR）等下游任务。

## 📌 项目特性
- **模型集成**：将 Qwen-Omni 接入为 s3prl 的标准 Upstream 模块。
- **在线推理**：提供 `expert_online.py` 专家模块，支持在线或流式特征提取逻辑。
- **预置任务**：包含针对 LibriSpeech (100h/960h)、AISHELL-1 及 SEAME 等数据集的 ASR 下游配置。

---

## 🛠 环境安装

建议使用 Conda 创建独立虚拟环境以确保依赖环境隔离：

```bash
# 1. 创建并激活环境
conda create -n speech_encoder python=3.10
conda activate speech_encoder

# 2. 安装 s3prl
pip install s3prl

# 3. 其他依赖
# 请确保环境中已安装运行 Qwen-Omni 架构所需的依赖库（如 transformers 等）
