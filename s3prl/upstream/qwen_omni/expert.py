# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""




import logging  # 日志记录
from pathlib import Path  

import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # 包含 layer_norm 等函数
from torch.nn.utils.rnn import pad_sequence  # 将不等长序列填充成等长 batch

from ..interfaces import UpstreamBase  # S3PRL 定义的上游模型基类
from .convert import load_converted_model  # 加载转换后的 HuBERT 模型

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)



# HuBERT 封装类，不依赖 fairseq
# 继承自 UpstreamBase，这是 S3PRL 的统一接口
class UpstreamExpert(UpstreamBase):  
    # 模型加载，注册了一系列 hooks（钩子），用于捕获每一层 encoder 的中间输出
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
       
        # 调用父类 UpstreamBase 的初始化方法
        model, task_cfg = load_converted_model(ckpt)
        self.model = model
        self.task_cfg = task_cfg
        
        # 加载预训练的 HuBERT 模型和其配置
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0
        
        # Hooks 注册部分，每一层 Transformer encoder 注册一个 hook
        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])
            
            # 为整个 encoder 注册 hook，捕获最终输出
            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess
   
    # 返回 320，表示每 320 个采样点输出 1 帧
    def get_downsample_rates(self, key: str) -> int:
        return 320

    # 核心逻辑：音频 → 特征
    def forward(self, wavs,**kwargs):
        print(wavs)
        sr = kwargs.get("sr",16000)
        index = self.support_sr.index(sr)
        assert index >= 0

        # 获取采样率并找到对应的索引，获取设备
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)

        # 构建 padding mask
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        # 将不等长的音频用 0 填充成相同长度的 batch
        features, layer_results = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
            index=index,
        )
       
        # 各层的隐藏状态
        return {
            "hidden_states": layer_results
        }
        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks




# 依赖 fairseq 的旧版本
class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.task = task

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks


class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.model.feature_grad_mult = 0.0
        self.task = task

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

        self._init_layerdrop = self.model.encoder.layerdrop

    @property
    def layer_drop(self):
        return self.model.encoder.layerdrop

    def set_layer_drop(self, layerdrop: float = None):
        if isinstance(layerdrop, float):
            self.model.encoder.layerdrop = layerdrop
        elif layerdrop is None:
            self.model.encoder.layerdrop = self._init_layerdrop
        else:
            raise ValueError("layerdrop can only be float or None")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
