# Copyright (c) Facebook, Inc. All Rights Reserved
#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert_online.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""

import os
import logging
logging.getLogger().setLevel(logging.ERROR) 
import warnings
warnings.filterwarnings("ignore", message="System prompt modified")
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

import numpy as np
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    # ============= 在线推理版本的 Qwen2.5-Omni 特征提取器 ============= 
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        
        # 设备选择
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(f"[QwenLite] Loading from: {ckpt}")
        logger.info(f"[QwenLite] Device: {self._device}")
        
        # 记录初始显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            init_mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[QwenLite] Initial GPU memory: {init_mem:.2f} GB")
        
        # 加载模型
        self._load_model(ckpt, self._device)

        # 最终显存
        if torch.cuda.is_available():
            final_mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[QwenLite] Final GPU memory: {final_mem:.2f} GB")


    # 加载模型，只保留 audio_tower + thinker.model
    def _load_model(self, model_path: str, device: str):
        
        # 加载完整模型
        logger.info("[QwenLite] Loading full model...")
        full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,   # 模型所在目录
            torch_dtype=torch.bfloat16,  # 把模型权重用 bfloat16 加载
            device_map=device,   # 告诉 Transformers 把模型放到哪里
            local_files_only=True,    # 禁止联网下载
        )
        
        # 保留 thinker 和 processor
        self.thinker = full_model.thinker
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
        
        # 获取模型thinker配置
        self._num_layers = len(self.thinker.model.layers)
        self._output_dim = self.thinker.model.config.hidden_size

        self._downsample_rate = 160
        logger.info(f"[QwenLite] Detected: {self._num_layers} layers, dim={self._output_dim}")
        
        # ========== 【改动】获取音频token的ID ==========
        self._audio_token_id = int(self.thinker.config.audio_token_id)
        logger.info(f"[QwenLite] Audio token ID: {self._audio_token_id}")

        # 删除不需要的模块
        logger.info("[QwenLite] Removing unnecessary modules...")
        
        # 删除视觉编码器
        if hasattr(self.thinker, 'visual'):
            del self.thinker.visual
            logger.info("[QwenLite]   - Deleted: visual")

        # 删除语音合成模块
        if hasattr(full_model, 'talker'):
            del full_model.talker
            logger.info("[QwenLite]   - Deleted: talker")
            
        # 删除# Token转波形模块
        if hasattr(full_model, 'token2wav'):
            del full_model.token2wav
            logger.info("[QwenLite]   - Deleted: token2wav")

        # 删除完整模型引用，释放显存
        del full_model
        torch.cuda.empty_cache()
        
        # 冻结参数，设为评估模式（关闭 dropout 等），不计算梯度，不更新参数，冻结Qwen 模型 (thinker)
        self.thinker.eval()
        for param in self.thinker.parameters():
            param.requires_grad = False

        logger.info("[QwenLite] Model loaded!")
    
    

    # ========== 下采样率 ==========
    # 每 640 个音频采样点 → 1 帧特征
    def get_downsample_rates(self, key: str) -> int:
        return self._downsample_rate
    
    # ========== 输出维度 ==========
    # 特征维度：7B=3584, 3B=2048
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    # 特征维度：7B=3584, 3B=2048
    @property
    def hidden_dim(self) -> int:
        return self._output_dim
    
    # 层数：7B=28, 3B=36
    @property
    def num_layers(self) -> int:
        return self._num_layers
    
    
    # ============= Featurizer 初始化时会用随机音频探测 upstream 的输出维度，这时候没有 preprocessed_inputs，所以 else 分支实际上是必需的=============
    def _preprocess_wavs(self, wavs: List[torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        all_conversations = []
        for wav in wavs:
            # 转换为 numpy
            if isinstance(wav, torch.Tensor):
                audio_np = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
            elif isinstance(wav, np.ndarray):
                audio_np = wav
            else:
                audio_np = np.array(wav)
            
            # 确保是 1D
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)
            
            # 构建对话格式
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a speech recognition model."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_np},
                        {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."},
                    ],
                }
            ]
            all_conversations.append(conversation)

        batch_inputs = self.processor.apply_chat_template(
            all_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            audio_kwargs={"sampling_rate": SAMPLE_RATE},
        )
   
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
    
    
    # ============= 核心 forward 方法 =============
    # 前向传播：音频 → 多层特征
    @torch.no_grad()
    def forward(
        self, 
        wavs: List[torch.Tensor],   # 输入：音频列表
        preprocessed_inputs: Optional[Dict[str, torch.Tensor]] = None,
        wav_paths: Optional[List[str]] = None,  # 兼容 cached 接口，但不使用
        **kwargs
    ) -> Dict[str, List[torch.Tensor]]:  # 输出：字典
        

        # 日志中屏蔽 Warning 警告
        import logging as _logging
        _root_logger = _logging.getLogger()
        _original_level = _root_logger.level
        _root_logger.setLevel(_logging.CRITICAL)

        try:
            # 获取模型所在设备
            device = next(self.thinker.parameters()).device
            if preprocessed_inputs is not None:
                # 优先使用 dataset.py 预处理好的数据（正常训练/评估流程）
                batch_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in preprocessed_inputs.items()
                }
            else:
                # 自行预处理（用于 Featurizer 初始化探测）
                batch_inputs = self._preprocess_wavs(wavs, device)

            # 模型前向传播，关键返回所有层的隐藏状态
            outputs = self.thinker(**batch_inputs, output_hidden_states=True, return_dict=True)

            # 提取 hidden states 28层 (跳过 embedding 层)
            hidden_states = outputs.hidden_states[1:self._num_layers + 1]
            
            # 转置: tensor(1, seq_len, dim) -> tensor(batch, max_seq_len, dim)
            layer_wise_features = [layer.float() for layer in hidden_states]
            

            #========== 【改动】提取纯音频token ==========
            input_ids = batch_inputs['input_ids']  # [batch, seq_len]
            batch_size = input_ids.shape[0]
            
            # 获取每个样本的音频token索引
            batch_audio_indices = []
            for b in range(batch_size):
                ids = input_ids[b].tolist()
                audio_indices = [i for i, tid in enumerate(ids) if tid == self._audio_token_id]
                batch_audio_indices.append(audio_indices)
            
            # 找到最大音频token数量（用于padding）
            max_audio_len = max(len(indices) for indices in batch_audio_indices)
            
            # 为每一层提取纯音频部分并padding
            audio_only_features = []
            for layer_hs in layer_wise_features:
                # layer_hs: [batch, seq_len, hidden_dim]
                batch_audio_tensors = []
                for b, audio_indices in enumerate(batch_audio_indices):
                    if len(audio_indices) > 0:
                        # 提取该样本的音频token
                        audio_hs = layer_hs[b, audio_indices, :]  # [num_audio_tokens, hidden_dim]
                        # Padding到最大长度
                        if len(audio_indices) < max_audio_len:
                            pad_len = max_audio_len - len(audio_indices)
                            padding = torch.zeros(pad_len, self._output_dim, device=device, dtype=audio_hs.dtype)
                            audio_hs = torch.cat([audio_hs, padding], dim=0)
                    else:
                        # 没有音频token（不应该发生）
                        audio_hs = torch.zeros(max_audio_len, self._output_dim, device=device, dtype=layer_hs.dtype)
                    batch_audio_tensors.append(audio_hs)
                
                # Stack成batch: [batch, max_audio_len, hidden_dim]
                layer_audio = torch.stack(batch_audio_tensors, dim=0)
                audio_only_features.append(layer_audio)
            # 保存结果
            result = {
                "hidden_states": audio_only_features,
                "last_hidden_state": audio_only_features[-1],
                }
                        
            # # 保存结果
            # result = {
            #     "hidden_states": layer_wise_features,
            #     "last_hidden_state": layer_wise_features[-1],
            #     }

        finally:
            _root_logger.setLevel(_original_level)
            
        return result

