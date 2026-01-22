#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    1. 从 QWEN2.5-Omni 提取全部 28 层隐层特征
    2. 权重正态分布初始化 N(0, 0.1)
"""


import os
import argparse
import logging 
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio


# ========== 抑制 WARNING ==========
logging.getLogger().setLevel(logging.ERROR)  # 只显示 ERROR 及以上
warnings.filterwarnings("ignore", message="System prompt modified")


# ============= 工具函数 =============

# 获取指定 split 下的所有 flac 文件
def get_flac(libri_root: str, split: str):
    split_dir = Path(libri_root) / split
    if not split_dir.exists():
        print(f"[WARNING] Split directory not found: {split_dir}")
        return []    
    flac_files = list(split_dir.rglob("*.flac"))
    print(f"[INFO] Found {len(flac_files)} files in {split}")
    return flac_files

# 加载 Qwen2.5-Omni 模型
def load_model(model_path: str, device: str):
    
    # 加载 Qwen2.5-Omni 模型和处理器
    from transformers import (Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor)
    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] Using device: {device}")
    
    # 加载模型，模型参数以 bfloat16格式加载/转换，local_files_only=True只从本地缓存/本地目录加载模型和处理器
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        device_map=device,
        local_files_only=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    
    # 设置模型为评估模式并冻结参数，确保了推理时行为稳定且不计算梯度
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print("[INFO] Model loaded successfully!")
    return model, processor


# ============= 提取全部28层 hidden states ============= 

# 禁用梯度计算，节省内存和计算资源
@torch.no_grad()
def extract_all_layers(model, processor, audio_path: str, device: str):

    # 加载音频，确保采样率为16kHz，单声道处理
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    audio_np = wav.numpy()
    
    # 构建对话格式，并 inputs 转换成模型可直接接受的输入格式，确保输入与模型设备一致
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
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True, audio_kwargs={"sampling_rate": 16000},)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    
    # 通过 thinker forward 获取所有 hidden states
    outputs = model.thinker(**inputs, output_hidden_states=True, return_dict=True)

    # 检测 hidden states 层数(优先 model.config, 其次 thinker.model.layers, 最后指定 28)
    num_layers = getattr(model.config, 'num_hidden_layers', None)
    if num_layers is None:
        try:
            num_layers = len(model.thinker.model.layers)
        except Exception:
            num_layers = 28
    # print(f"len(outputs.hidden_states): {len(outputs.hidden_states)}")
    # print(f"Detected decoder num_layers = {num_layers}")

    # 提取全部 num_layers 层 hidden states, 跳过第0层embedding
    all_hidden = outputs.hidden_states[1:1 + num_layers]

    # 拼接成 ndarray (num_layers, 1, seq_len, hidden_dim), 转为 float16 节省空间
    stacked = torch.stack(all_hidden, dim=0)
    multilayer_np = stacked.cpu().to(torch.float16).numpy()

    # hidden_states_multilayer：NumPy 数组，shape=(num_layers, batch, seq_len, hidden_dim)，dtype float16；
    return {
        'hidden_states_multilayer': multilayer_np,
    }


# ============= 处理一个 split 的所有音频，使用多层模式 ============= 
def process_split(model, processor, libri_root: str, split: str, output_root: str, device: str, skip_existing: bool = True):
    
    # 调用函数 get_flac 获取所有 flac 文件
    flac_files = get_flac(libri_root, split)
    
    # hidden 输出目录
    split_name = split.replace('-', '_')
    output_dir = Path(output_root) / f"lib_hidden_{split_name}"
    
    # processed：已经处理的, skipped：跳过的, failed：失败的
    processed = 0
    skipped = 0
    failed = 0

    # 逐个 flac 文件处理
    for flac_path in tqdm(flac_files, desc=f"Extracting {split}"):
        try:
            relative = flac_path.relative_to(Path(libri_root) / split)
            parts = relative.parts
            speaker = parts[0]
            chapter = parts[1]
            filename = flac_path.stem
            feat_dir = output_dir / speaker / chapter
            feat_path = feat_dir / f"{filename}_seq.npz"
            if skip_existing and feat_path.exists():
                skipped += 1
                continue
            feat_dir.mkdir(parents=True, exist_ok=True)

            # 调用函数 extract_all_layers 多层提取, 并保存为 NPZ 文件
            feat_dict = extract_all_layers(model, processor, str(flac_path), device)
            np.savez(feat_path, **feat_dict)
            processed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {flac_path}: {e}")
            failed += 1
            continue
    print(f"\n[INFO] {split} completed: processed={processed}, skipped={skipped}, failed={failed}")


# ============= 主函数 =============
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/s2025244189/s1025244003/Qwen2.5-Omni-7B", help="QWEN root")
    parser.add_argument("--libri_root", type=str, default="/home/s2025244189/datas/LibriSpeech", help="LibriSpeech root")
    parser.add_argument("--output_root", type=str, default="/home/s2025244189/s1025244003/s3prl/s3prl/data/hidden_layer", help="output root")
    parser.add_argument("--split", type=str, default="all", help="split (all/train/dev/test/)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU num")
    parser.add_argument("--no_skip", action="store_true", help="不跳过已存在的文件，重新处理")
    args = parser.parse_args()
    
    # 设置设备
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 加载模型，调用函数load_model
    model, processor = load_model(args.model_path, device)
    
    # 处理指定 split (数据集)
    all_splits = ['train-clean-100','train-clean-360','train-other-500','dev-clean', 'dev-other', 'test-clean', 'test-other']
    if args.split == "all":
        splits = all_splits
    elif args.split == "train":
        splits = ['train-clean-100','train-clean-360','train-other-500']
    elif args.split == "dev":
        splits = ['dev-clean', 'dev-other']
    elif args.split == "test":
        splits = ['test-clean', 'test-other']
    else:
        splits = [args.split]
    
    # 调用函数process_split，逐个 split 处理，并打印输出
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing: {split}")
        print(f"{'='*60}")
        process_split(model, processor, args.libri_root, split, args.output_root, device, skip_existing=not args.no_skip)
    print("\n[INFO] All done!")

if __name__ == "__main__":
    main()
