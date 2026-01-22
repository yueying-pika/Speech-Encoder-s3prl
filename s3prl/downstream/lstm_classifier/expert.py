#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Downstream Expert for S3PRL

一个简单的 LSTM 分类器示例，展示如何在 S3PRL 框架中
使用 Qwen2.5-Omni upstream 的输出来训练下游任务。

任务类型：Utterance-level Classification（如说话人识别、情感识别等）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random


class LSTMClassifier(nn.Module):
    """
    简单的 LSTM 分类器
    
    输入：upstream 的 hidden states [batch, seq_len, hidden_dim]
    输出：分类 logits [batch, num_classes]
    """
    def __init__(
        self,
        input_dim: int,        # upstream 输出维度（如 Qwen 的 3584）
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, features, feature_lens=None):
        """
        Args:
            features: [batch, seq_len, input_dim]
            feature_lens: [batch] 每条序列的实际长度
        
        Returns:
            logits: [batch, num_classes]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(features)
        # lstm_out: [batch, seq_len, hidden_dim * num_directions]
        
        # 使用最后一个时间步的输出（或者可以用 mean pooling）
        if feature_lens is not None:
            # 取每条序列最后一个有效位置的输出
            batch_size = features.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_idx = feature_lens[i] - 1
                last_outputs.append(lstm_out[i, last_idx, :])
            pooled = torch.stack(last_outputs, dim=0)
        else:
            # Mean pooling
            pooled = lstm_out.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits


class SimpleDataset(Dataset):
    """
    简单的数据集示例
    实际使用时请替换为你的真实数据集
    """
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 这里应该加载你的真实数据
        # 格式：[(audio_path, label), ...]
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """加载数据样本，请根据你的数据格式修改"""
        # 示例：假设有一个 metadata.txt 文件
        # 每行格式：audio_path label
        samples = []
        
        metadata_file = self.data_dir / f"{self.split}.txt"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        audio_path = parts[0]
                        label = int(parts[1])
                        samples.append((audio_path, label))
        else:
            # 如果没有 metadata，生成一些假数据用于测试
            print(f"[Warning] {metadata_file} not found, using dummy data")
            for i in range(100):
                samples.append((f"dummy_{i}.wav", i % 10))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # 加载音频
        # 实际使用时应该用 torchaudio.load
        # 这里返回假数据用于测试
        import torchaudio
        
        full_path = self.data_dir / audio_path
        if full_path.exists():
            wav, sr = torchaudio.load(str(full_path))
            if wav.dim() == 2:
                wav = wav.mean(dim=0)  # 转单声道
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
        else:
            # 假数据
            wav = torch.randn(16000 * 3)  # 3秒
        
        return wav, label


def collate_fn(batch):
    """
    自定义 collate function
    S3PRL 的 upstream 期望输入是 list of wavs
    """
    wavs = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    return wavs, labels


class DownstreamExpert(nn.Module):
    """
    S3PRL Downstream Expert
    
    必须实现的方法：
    - get_dataloader(split): 返回 DataLoader
    - forward(split, features, ...): 前向传播，返回 loss
    - log_records(split, records, ...): 记录日志
    """
    
    def __init__(
        self,
        upstream_dim: int,      # upstream 输出维度
        upstream_rate: int,     # upstream 下采样率
        datarc: dict,           # 数据配置
        modelrc: dict,          # 模型配置
        **kwargs
    ):
        super().__init__()
        
        self.upstream_dim = upstream_dim
        self.datarc = datarc
        self.modelrc = modelrc
        
        # 创建 LSTM 分类器
        self.model = LSTMClassifier(
            input_dim=upstream_dim,
            hidden_dim=modelrc.get("hidden_dim", 256),
            num_layers=modelrc.get("num_layers", 2),
            num_classes=modelrc.get("num_classes", 10),
            dropout=modelrc.get("dropout", 0.1),
            bidirectional=modelrc.get("bidirectional", True),
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录
        self.records = {}
    
    def get_dataloader(self, split: str) -> DataLoader:
        """
        返回指定 split 的 DataLoader
        
        Args:
            split: "train", "dev", 或 "test"
        
        Returns:
            DataLoader
        """
        dataset = SimpleDataset(
            data_dir=self.datarc.get("data_dir", "./data"),
            split=split,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.datarc.get("batch_size", 8),
            shuffle=(split == "train"),
            num_workers=self.datarc.get("num_workers", 4),
            collate_fn=collate_fn,
            drop_last=(split == "train"),
        )
    
    def forward(
        self,
        split: str,
        features: torch.Tensor,
        labels: torch.Tensor,
        feature_lens: torch.Tensor = None,
        records: dict = None,
        **kwargs
    ):
        """
        前向传播
        
        Args:
            split: "train", "dev", 或 "test"
            features: [batch, seq_len, hidden_dim] upstream 的输出
            labels: [batch] 标签
            feature_lens: [batch] 序列长度
            records: 用于记录指标的 dict
        
        Returns:
            loss: 标量 loss
        """
        # LSTM forward
        logits = self.model(features, feature_lens)
        
        # 计算 loss
        loss = self.criterion(logits, labels)
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        
        # 记录
        if records is not None:
            records["loss"].append(loss.item())
            records["acc"].append(acc)
        
        return loss
    
    def log_records(
        self,
        split: str,
        records: dict,
        logger,
        global_step: int,
        batch_ids: list = None,
        total_batch_num: int = None,
        **kwargs
    ):
        """
        记录训练/验证日志
        
        Args:
            split: "train", "dev", 或 "test"
            records: forward 中记录的指标
            logger: tensorboard logger
            global_step: 当前步数
        """
        avg_loss = sum(records["loss"]) / len(records["loss"])
        avg_acc = sum(records["acc"]) / len(records["acc"])
        
        # 打印
        print(f"[{split}] Step {global_step}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        # TensorBoard
        if logger is not None:
            logger.add_scalar(f"{split}/loss", avg_loss, global_step)
            logger.add_scalar(f"{split}/accuracy", avg_acc, global_step)
        
        # 返回用于 early stopping 的指标
        return {"loss": avg_loss, "acc": avg_acc}
