# Qwen2.5-Omni Upstream for S3PRL

将 Qwen2.5-Omni 的 Audio Encoder + Thinker 作为 S3PRL 的 upstream，用于提取语音表示。

## 🚀 快速开始

### 1. 安装文件

将整个 `qwen_omni` 文件夹复制到 s3prl 的 upstream 目录：

```bash
# 假设你的 s3prl 安装在 /path/to/s3prl
cp -r qwen_omni /path/to/s3prl/s3prl/upstream/
```

目录结构应该是：
```
s3prl/
└── s3prl/
    └── upstream/
        └── qwen_omni/
            ├── __init__.py
            ├── expert.py
            ├── hubconf.py
            └── README.md
```

### 2. 注册 Upstream

编辑 `s3prl/hub.py`，在文件末尾添加：

```python
from s3prl.upstream.qwen_omni.hubconf import *
```

### 3. 修改模型路径

编辑 `expert.py` 或 `hubconf.py`，将模型路径改为你的实际路径：

```python
# 在 hubconf.py 的 qwen_omni_7b 函数中
default_path = "/your/path/to/Qwen2.5-Omni-7B"
```

### 4. 验证安装

```bash
cd /path/to/s3prl/s3prl

# 测试能否正确加载
python3 -c "
from s3prl.hub import qwen_omni_7b
model = qwen_omni_7b()
print('✓ Upstream loaded successfully!')
print(f'  Output dim: {model.output_dim}')
"
```

### 5. 运行 Downstream 训练

```bash
# 使用预设路径的 qwen_omni_7b
python3 run_downstream.py -m train \
    -u qwen_omni_7b \
    -d example \
    -n QwenOmni_Test

# 或者使用自定义路径
python3 run_downstream.py -m train \
    -u qwen_omni_local \
    -k /your/path/to/Qwen2.5-Omni-7B \
    -d example \
    -n QwenOmni_Test
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `expert.py` | 核心实现，包含 `UpstreamExpert` 类 |
| `hubconf.py` | 入口注册，定义可用的 upstream 名称 |
| `__init__.py` | 模块初始化 |

## 🔧 关键参数

### UpstreamExpert 属性

| 属性 | 值 | 说明 |
|------|-----|------|
| `output_dim` | 3584 | Thinker 输出的 hidden dimension |
| `downsample_rate` | 640 | 16kHz 音频到特征帧的下采样率（近似值） |

### 入口函数

| 入口名 | 说明 | 用法 |
|--------|------|------|
| `qwen_omni_7b` | 预设本地路径 | `-u qwen_omni_7b` |
| `qwen_omni_local` | 自定义本地路径 | `-u qwen_omni_local -k /path` |
| `qwen_omni_custom` | 通用路径 | `-u qwen_omni_custom -k path_or_name` |
| `qwen_omni` | HuggingFace 下载 | `-u qwen_omni` |

## ⚠️ 注意事项

### 1. 显存需求
Qwen2.5-Omni-7B 需要约 **16GB+ 显存**。建议使用：
- 单卡 A100 (40GB/80GB)
- 单卡 A6000 (48GB)
- 多卡时指定 `device_map`

### 2. 下采样率
`downsample_rate` 是一个近似值。由于 Qwen2.5-Omni 使用 tokenization，
实际的序列长度可能不完全符合这个比率。对于 **utterance-level** 任务（如说话人识别、
情感识别）影响较小；对于 **frame-level** 任务（如 ASR、diarization）可能需要调整。

### 3. 冻结参数
默认情况下，所有 Qwen 参数都被冻结（`requires_grad=False`）。
如果需要微调，可以修改 `expert.py` 中的代码。

### 4. 处理速度
由于每条音频都需要经过完整的 thinker forward，处理速度较慢。
建议：
- 预提取特征保存到磁盘
- 或者只使用 audio_tower 而不是完整的 thinker（需要修改代码）

## 🔬 进阶用法

### 只使用 Audio Encoder（不用 Thinker）

如果你只想要 audio encoder 的输出（维度 1280），可以修改 `expert.py`：

```python
# 在 _extract_single 方法中，改为：
audio_features = self.processor(
    audios=audio_np, 
    return_tensors="pt", 
    sampling_rate=16000
)["input_features"].to(device)

hidden_states = self.model.thinker.audio_tower(audio_features)
# 输出维度会变成 1280（经过 proj 层后是 3584）
```

### 自定义 Downstream（LSTM 示例）

参见下方的 `downstream_lstm_example/` 目录。

## 📝 参考

- [S3PRL Documentation](https://s3prl.github.io/s3prl/)
- [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [Adding New Upstream](https://s3prl.github.io/s3prl/contribute/upstream.html)
