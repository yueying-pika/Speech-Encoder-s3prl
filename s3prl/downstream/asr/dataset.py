# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############

# 屏蔽警告
import warnings
import logging
warnings.filterwarnings("ignore", message="System prompt modified")
logging.getLogger().setLevel(logging.ERROR)


import logging
import os
import random
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
from .dictionary import Dictionary

# 使用soundfile代替torchaudio
import soundfile
from scipy.signal import resample_poly
import numpy as np


from transformers import Qwen2_5OmniProcessor

SAMPLE_RATE = 16000  # 音频采样率 16kHz
HALF_BATCHSIZE_TIME = 2000  # 如果音频长度超过 2000 帧，batch size 减半（防止 OOM）


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    
    def __init__(self, split, bucket_size, dictionary, libri_root, bucket_file, **kwargs):
        super(SequenceDataset, self).__init__()
        
        # 保存基本配置
        self.dictionary = dictionary
        self.libri_root = libri_root
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
        
        # 延迟初始化
        self.processor = None
        self.model_path = kwargs.get('qwen_model_path', '/home/s2025244189/s1025244003/Qwen2.5-Omni-7B')
        
        # 检查 bucket 文件目录是否存在
        assert os.path.isdir(bucket_file), 'Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file.'

        # 读取每个子集的 CSV 文件
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_file, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )
            else:
                logging.warning(f'{item} is not found in bucket_file: {bucket_file}, skipping it.')
        
        # 合并所有 CSV，按长度降序排序
        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)
        
        # 提取文件路径和长度列表，并检查数据不为空
        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()
        assert len(X) != 0, f"0 data found for {split}"

        # 加载所有音频对应的文本标注
        Y = self._load_transcript(X)

        # 计算音频文件名和标注文件名的交集，确保每个音频都有对应的标注
        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)
        Y = {key: Y[key] for key in usage_list}
        self.Y = {
            k: self.dictionary.encode_line(
                v, line_tokenizer=lambda x: x.split()
            ).long() 
            for k, v in Y.items()
        }

        # 使用bucketing策略防止padding过多OOM
        self.X = []
        batch_x, batch_len = [], []
        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'ASR dataset {split}', dynamic_ncols=True):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)
    
    # 延迟初始化模式
    def _get_processor(self):
        if self.processor is None:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
        return self.processor
    
    # 从路径提取文件名
    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    # soundfile加载方式，重试三次，不行跳过
    def _load_wav(self, wav_path, retries=3):
        import time
        full_path = os.path.join(self.libri_root, wav_path)
        for attempt in range(retries):
            try:
                data, sr = soundfile.read(full_path, dtype='float32')
                if sr != SAMPLE_RATE:
                    gcd = np.gcd(SAMPLE_RATE, sr)
                    data = resample_poly(data, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
                wav = torch.from_numpy(data).float()
                if wav.dim() > 1:
                    wav = wav.mean(dim=-1)
                return wav
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.1)
                    continue
                print(f"[SKIP] {wav_path}: {e}")  # 打印路径
                return None  # 跳过，不崩溃
        
    # 文本预处理函数
    def _load_transcript(self, x_list):
        def process_trans(transcript):
            #TODO: support character / bpe
            transcript = transcript.upper()
            return " ".join(list(transcript.replace(" ", "|"))) + " |"
        trsp_sequences = {}
        split_spkr_chap_list = list(
            set(
                "/".join(x.split('/')[:-1]) for x in x_list
            )
        )
        for dir in split_spkr_chap_list:
            parts = dir.split('/')
            trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
            
            # 处理缓存特征目录名到原始LibriSpeech目录名的映射
            mapped_dir = dir
            name_mapping = {
                'lib_hidden_test_clean': 'test-clean',
                'lib_hidden_test_other': 'test-other',
                'lib_hidden_dev_clean': 'dev-clean',
                'lib_hidden_dev_other': 'dev-other',
                'lib_hidden_train_clean_100': 'train-clean-100',
            }
            for old_name, new_name in name_mapping.items():
                if old_name in mapped_dir:
                    mapped_dir = mapped_dir.replace(old_name, new_name)
                    break           
            path = os.path.join(self.libri_root, mapped_dir, trans_path)
            assert os.path.exists(path), f"Transcript not found: {path}"
            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))
        return trsp_sequences

    # 构建词典
    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d
   
    # 在 worker 中预处理音频
    def _preprocess_audio(self, audio_np):
        processor = self._get_processor()
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
        
        # 处理单条数据，不做 batch
        inputs = processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,  # 单条不需要 padding
            audio_kwargs={"sampling_rate": SAMPLE_RATE},
        )  
        return inputs
    
    # 返回 batch 数量
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        wav_batch = []
        processed_batch = []
        label_batch = []
        filename_batch = []
        filepath_batch = []
        
        for x_file in self.X[index]:
            # 加载音频

            wav = self._load_wav(x_file)
            if wav is None:
                continue  # 新增，跳过失败的
            wav_np = wav.numpy()
            processed = self._preprocess_audio(wav_np)
            wav_batch.append(wav)
            processed_batch.append(processed)
            label_batch.append(self.Y[self._parse_x_name(x_file)].numpy())
            filename_batch.append(Path(x_file).stem)
            filepath_batch.append(x_file)
        
        # 新增：如果整个 batch 都失败，取下一个 batch
        # if len(wav_batch) == 0:
        #     return self.__getitem__((index + 1) % len(self))
    
        return wav_batch, processed_batch, label_batch, filename_batch, filepath_batch

    def collate_fn(self, items):
        assert len(items) == 1
        wav_batch, processed_batch, label_batch, filename_batch, filepath_batch = items[0]
        
        # 合并预处理后的 inputs
        merged_inputs = self._merge_processed_inputs(processed_batch)
        
        return wav_batch, merged_inputs, label_batch, filename_batch, filepath_batch

    def _merge_processed_inputs(self, processed_batch):
        if len(processed_batch) == 1:
            return processed_batch[0]
        
        # 找出所有的 key
        keys = processed_batch[0].keys()
        merged = {}
        
        for key in keys:
            tensors = [p[key].squeeze(0) for p in processed_batch]  # 去掉 batch 维度
            
            if tensors[0].dim() == 0:
                # 标量，直接 stack
                merged[key] = torch.stack(tensors)
            elif key in ['input_ids', 'attention_mask']:
                # 需要 padding
                merged[key] = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=0
                )
            else:
                # 其他 tensor，尝试 stack 或 pad
                try:
                    merged[key] = torch.stack(tensors)
                except:
                    merged[key] = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=0
                    )
        
        return merged