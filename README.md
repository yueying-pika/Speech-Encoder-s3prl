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
```

---

## 📂 项目结构说明 (Project Structure)

```text
Speech-Encoder-s3prl/
├── s3prl/
│   ├── upstream/
│   │   └── qwen_omni/              # Upstream 核心实现目录
│   │       ├── expert_online.py    # Qwen-Omni 专家类实现 (支持 Online/Stream)
│   │       ├── run_qwen_online.sbatch   # 训练启动脚本 (SLURM)
│   │       └── evaluate.sbatch          # 评估启动脚本 (SLURM)
│   └── downstream/
│       ├── asr/                    # LibriSpeech ASR 任务配置文件
│       │   ├── config_qwen_omni_100h_online.yaml
│       │   └── config_qwen_omni_960h_online.yaml
│       ├── asr_aishell_1/          # 中文 AISHELL-1 任务支持
│       └── asr_seame/              # 中英混读 SEAME 任务支持

---

## 🚀 快速开始
1. 训练脚本 (Training)
使用 SLURM 系统提交训练作业，或参考脚本内的指令进行本地运行：
# 提交 ASR 训练任务（默认使用 S3PRL runner）
sbatch s3prl/upstream/qwen_omni/run_qwen_online.sbatch

2. 评估脚本 (Evaluation)
对训练好的 Checkpoint 进行性能评估（如计算 WER 等指标）：
# 提交评估任务
sbatch s3prl/upstream/qwen_omni/evaluate.sbatch

---

## 🔗 参考资料
s3prl 官方仓库
Qwen 系列模型说明
