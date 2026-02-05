import torch
import lightning.pytorch as pl
import torch.nn as nn
import soundfile as sf
import torchaudio
from typing import List, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel, EncDecMaskedTokenPredModel
from nemo.collections.asr.models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
import pandas as pd


def save_tensor_to_csv(tensor, path, name):
    """
    tensor: torch.Tensor, shape (1, T, D) 或 (1, D, T)
    """
    tensor = tensor.detach().cpu()

    # 去掉 batch 维
    tensor = tensor.squeeze(0)
    df = pd.DataFrame(tensor.numpy())
    df.to_csv(path, index=False)
    print(f"{name} saved to {path}, shape = {df.shape}")

def load_wav_ta(wav_path):
  #original implementation.
    wav, sr = torchaudio.load(wav_path)
    wav = wav.squeeze(0)
    print(
        "================[WAV CHECK]================\n",
        f"wav.min().item() = {wav.min().item()}\n",
        f"wav.max().item() = {wav.max().item()}\n",
        f"wav.mean().item() = {wav.mean().item()}\n",
        f"wav.std().item() = {wav.std().item()}\n"
        f"wav.shape = {wav.shape}\n",
        "===========================================",
    )
    wav_len = wav.shape[0] 
    return wav,wav_len

def load_wav_sf(wav_path):
    wav, sr = sf.read(wav_path, dtype='float32')

    wav = torch.from_numpy(wav)
    print(
        "================[WAV CHECK]================\n",
        f"wav.min().item() = {wav.min().item()}\n",
        f"wav.max().item() = {wav.max().item()}\n",
        f"wav.mean().item() = {wav.mean().item()}\n",
        f"wav.std().item() = {wav.std().item()}\n"
        f"wav.shape = {wav.shape}\n",
        "===========================================",
    )
    # 多通道 → 单通道
    if wav.dim() == 2:
        wav = wav.mean(dim=1)
    # print(f"wav.shape = {wav.shape}")
    wav_len = wav.shape[0] 
    return wav,wav_len


def main():
    device = "cpu"
    # =========================
    # 1. 加载参数
    # =========================
    cfg = OmegaConf.load(r"/Work21/2026/liangjintao/huaweieval/model_config.yaml")
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    model_params = torch.load(r"/Work21/2026/liangjintao/huaweieval/model_for_eval/SFT_50wh.pt", map_location='cpu', weights_only=False)['module'] 

    model = EncDecDenoiseMaskedTokenPredModel(cfg.model, trainer=trainer).eval()
    missing_keys, unexpected_keys = model.load_state_dict(model_params, strict=False)

    preprocessor = model.preprocessor
    encoder = model.encoder
   
    preprocessor.eval()
    encoder.eval()
    # =========================
    # 2. 读取单个 wav
    # =========================
    wav_path = "/Work21/2026/liangjintao/Datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    wav, wav_len = load_wav_sf(wav_path)
    wav = wav.unsqueeze(0)
    print(f"wav_len = {wav_len}")
    wav_len = torch.tensor([wav_len], device=device)
    # =========================
    # 3. preprocess & encode
    # =========================
    processed, processed_len = preprocessor(
        input_signal=wav,
        length=wav_len
    )
    print(f"processed_signal shape = {processed.shape}")
    print("processed content:", processed)
    with torch.no_grad():
        encoded, encoded_len = encoder(
            audio_signal=processed,
            length=processed_len
        )
    # =========================
    # 4. 打印encoder输出信息
    # =========================
    print("Encoder feature shape (B,D,T):", encoded.shape)
    print("Encoder feature content:", encoded)
    # =========================
    # 5. save as csv
    # =========================
    save_tensor_to_csv(processed, "processed.csv", "processed")
    save_tensor_to_csv(encoded, "encoded.csv", "encoded")

if __name__ == "__main__":
    main()
