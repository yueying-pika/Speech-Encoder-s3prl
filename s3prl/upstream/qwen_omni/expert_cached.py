# Copyright (c) Facebook, Inc. All Rights Reserved
#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/qwen/expert_cached.py ]
#   Synopsis     [ the QWEN wrapper ]
#   Author       [ Yueying Chen ]
"""*********************************************************************************************"""

"""
expert_cached.py - 从预提取的特征文件加载 Qwen hidden states

expert.py: 音频 → 模型实时推理 → 特征 → 下游任务  (注册hook, 捕获每一层 encoder 的中间输出, 采样率, padding, forward 提取 hidden)
                     ↑ 慢、重复计算

expert_cached.py: 音频 → [离线预提取] → NPZ文件 → 快速加载 → 下游任务   (加载hidden, 采样率, 匹配wav和wav_paths, padding, forward 对 hidden 格式转换)
                           ↑ 快、节省计算
"""


import os
import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Optional
from pathlib import Path


class UpstreamExpert(nn.Module):
    
    # ============= 从预提取的 .npz 文件加载 Qwen2.5-Omni 特征, 返回多层特征 list ============= 
    def __init__(self, ckpt: str = None, **kwargs):
        super().__init__()
    
        # 设备选择
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        print(f"[CachedUpstream] LOCAL_RANK={local_rank}, using device={self._device}")

        # 特征根目录
        self.feat_root = Path(ckpt)
        
        # split 名称到特征目录的映射
        self.split_mapping = {
            'train-clean-100': 'lib_hidden_train_clean_100',
            # 'train-clean-360': 'lib_hidden_train_clean_360',
            # 'train-other-500': 'lib_hidden_train_other_500',
            'dev-clean': 'lib_hidden_dev_clean',
            'dev-other': 'lib_hidden_dev_other',
            'test-clean': 'lib_hidden_test_clean',
            'test-other': 'lib_hidden_test_other',
        }
        
        # 多层特征配置
        self._output_dim = 3584 # Qwen2.5-Omni 的隐层维度
        self._downsample_rate = 640 # 音频下采样率
        self.use_multilayer = False
        self.num_layers = 28  # 默认 28 层
        
        print(f"[CachedUpstream] Feature root: {self.feat_root}")
        print(f"[CachedUpstream] Output dim: {self._output_dim}")
        print(f"[CachedUpstream] Using Featurizer weighted sum (S3PRL native)")
        
        # 容错配置
        self._max_retries = 3
        self._retry_delay = 0.5
        self._missing_file_count = 0
        self._max_missing_warnings = 10
        
        # 验证目录存在
        for split, feat_dir in self.split_mapping.items():
            full_path = self.feat_root / feat_dir
            if full_path.exists():
                print(f"[CachedUpstream] Found: {feat_dir}")
            else:
                print(f"[CachedUpstream] WARNING: Not found: {feat_dir}")
    


    # ============= 返回下游任务特征维度和下采样率 ============= 
    def get_downsample_rates(self, key: str) -> int:
        return self._downsample_rate
    @property
    def output_dim(self) -> int:
        return self._output_dim
    @property
    def hidden_dim(self) -> int:
        return self._output_dim
    


    # ============= 解析音频路径 ============= 
    # 输入: "train-clean-100/19/198/19-198-0000.flac"
    # 输出: ('train-clean-100', '19', '198', '19-198-0000')
    #        ↑ split        ↑ speaker ↑ chapter ↑ filename
    def _parse_wav_path(self, wav_path: str):
        wav_path = str(wav_path)
        path_obj = Path(wav_path)
        parts = path_obj.parts
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if part in self.split_mapping:
                remaining_parts = len(parts) - i - 1
                if remaining_parts >= 3:
                    return (
                        part,              # split
                        parts[i + 1],      # speaker  
                        parts[i + 2],      # chapter
                        path_obj.stem      # filename
                    )       
        raise ValueError(
            f"Cannot parse wav path: {wav_path}\n"
            f"  Path parts: {parts}\n"
            f"  Known splits: {list(self.split_mapping.keys())}"
        )
    

    # ============= 将音频路径转换为特征文件路径 ============= 
    def _get_feat_path(self, wav_path: str) -> Path:
        split, speaker, chapter, filename = self._parse_wav_path(wav_path)
        feat_dir = self.feat_root / self.split_mapping[split] / speaker / chapter
        feat_path = feat_dir / f"{filename}_seq.npz"
        return feat_path
    

    # ============= 加载特征文件，多层特征: (num_layers, seq_len, 3584) ============= 
    def _load_feature_with_retry(self, feat_path: Path, wav_path: str) -> Optional[torch.Tensor]:
        
        # 重试机制（处理 NFS 缓存问题）
        for attempt in range(self._max_retries):
            if feat_path.exists():
                try:
                    data = np.load(feat_path)
                    
                    # 检测是否为多层特征
                    if 'hidden_states_multilayer' in data:
                        # 多层特征模式
                        multilayer = data['hidden_states_multilayer']  # (num_layers, 1, seq_len, 3584)
                        
                        # 动态检测层数
                        actual_num_layers = multilayer.shape[0]
                        if not self.use_multilayer:
                            self.use_multilayer = True
                            self.num_layers = actual_num_layers
                            print(f"[CachedUpstream] Multi-layer features detected! Layers: {actual_num_layers}")
                            if 'layer_indices' in data:
                                print(f"[CachedUpstream] Layer indices: {data['layer_indices']}")
                        
                        # 移除 batch 维度：(num_layers, 1, seq_len, 3584) -> (num_layers, seq_len, 3584)
                        multilayer = multilayer.squeeze(1)
                        return torch.from_numpy(multilayer).float()
                    
                    else:
                        # 单层特征模式（向后兼容）
                        hidden = data['hidden_states']  # (1, seq_len, 3584)
                        return torch.from_numpy(hidden).float()
                    
                except Exception as e:
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_delay)
                        continue
                    else:
                        # 文件损坏
                        self._missing_file_count += 1
                        if self._missing_file_count <= self._max_missing_warnings:
                            print(f"[CachedUpstream] WARNING: Failed to load {feat_path}: {e}")
                        return None
            else:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
                    continue
        
        # 文件不存在
        self._missing_file_count += 1
        if self._missing_file_count <= self._max_missing_warnings:
            print(f"[CachedUpstream] WARNING: File not found after {self._max_retries} retries: {feat_path}")
            print(f"[CachedUpstream] Original path: {wav_path}")
        if self._missing_file_count == self._max_missing_warnings:
            print(f"[CachedUpstream] WARNING: Suppressing further missing file warnings...")
        
        return None


    #  ============= Forward 方法 ============= 
    # 输入: wav_paths = [path1, path2, path3]  # 假设batch=3
    # 输出: {"hidden_states": List[28个 Tensor(3, seq_len, 3584)]}
    def forward(self, wavs: List[torch.Tensor], wav_paths: Optional[List[str]] = None, **kwargs ) -> Dict[str, List[torch.Tensor]]: # wav_paths 特征文件路径，不是直接加载wav, 这是最大的改动

        # 设备选择逻辑
        if wavs and len(wavs) > 0 and isinstance(wavs[0], torch.Tensor) and wavs[0].is_cuda:
            device = wavs[0].device
        else:
            device = torch.device(self._device)

        if wav_paths is None:
            # 没有路径时返回 dummy 特征
            batch_size = len(wavs) if wavs else 1
            dummy_layers = []
            for _ in range(self.num_layers):
                dummy_hidden = torch.zeros(
                    batch_size, 10, self._output_dim,
                    dtype=torch.float32,
                    device=device
                )
                dummy_layers.append(dummy_hidden)
            
            return {
                "hidden_states": dummy_layers,  # List of tensors
                "last_hidden_state": dummy_layers[-1],
            }
        
        # 加载所有文件的特征
        all_layers = []
        for wav_path in wav_paths:
            feat_path = self._get_feat_path(wav_path)
            multilayer = self._load_feature_with_retry(feat_path, wav_path)
            if multilayer is None:
                if self.use_multilayer:
                    multilayer = torch.zeros(self.num_layers, 50, self._output_dim, dtype=torch.float32)
                else:
                    multilayer = torch.zeros(1, 50, self._output_dim, dtype=torch.float32)
            all_layers.append(multilayer)
        
        # 转置：从 [batch][layers, seq_len, dim] 到 [layers][batch, seq_len, dim]，
        num_layers = all_layers[0].size(0)
        batch_size = len(all_layers)
        
        # 每一层单独 padding
        layer_wise_features = []
        for layer_idx in range(num_layers):
            layer_features = [sample[layer_idx] for sample in all_layers]
            max_len = max(f.size(0) for f in layer_features)
            padded = torch.zeros(
                batch_size, max_len, self._output_dim,
                dtype=torch.float32,
                device=device
            )
            for i, f in enumerate(layer_features):
                seq_len = f.size(0)
                padded[i, :seq_len, :] = f.to(device)
            layer_wise_features.append(padded)
        
        # 返回
        return {
            "hidden_states": layer_wise_features, 
            "last_hidden_state": layer_wise_features[-1],
        }


if __name__ == "__main__":
    """测试代码"""
    upstream = UpstreamExpert()
    
    # 测试各种路径格式
    test_paths = [
        "train-clean-100/19/198/19-198-0000.flac",
    ]
    
    for test_path in test_paths:
        try:
            feat_path = upstream._get_feat_path(test_path)
            print(f"Audio: {test_path}")
            print(f"Feature: {feat_path}")
            print(f"Exists: {feat_path.exists()}")
            print()
        except ValueError as e:
            print(f"Error parsing: {test_path}")
            print(f"  {e}")
            print()