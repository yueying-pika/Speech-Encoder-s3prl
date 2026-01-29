import os
import re
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME_SEC = 20.0


@dataclass(frozen=True)
class AishellSegment:
    utt_id: str
    wav_path: str
    duration_sec: float
    text: str


_FIELD_RE = re.compile(r"^(?P<key>[a-zA-Z_]+):(?P<val>.*)$")


def _parse_kv_line(line: str) -> Dict[str, str]:
    """
    Parse `key:value` pairs from a manifest line.

    Supports BOTH formats:
    1) Tab separated kvs:
       utt:ID\\tfeat:/abs.wav\\tfeat_shape:3.21\\ttext:...
    2) Whitespace separated kvs:
       utt:ID feat:/abs.wav feat_shape:3.21 text:...

    Values may contain spaces (especially `text:`), so we parse by locating the
    key markers and slicing between them.
    """
    line = (line or "").strip()
    if not line:
        return {}

    # Fast path: explicit tab-separated kv tokens
    if "\t" in line:
        kv: Dict[str, str] = {}
        for p in line.split("\t"):
            m = _FIELD_RE.match(p.strip())
            if not m:
                continue
            kv[m.group("key")] = m.group("val").strip()
        return kv

    # Robust path: scan keys and slice values until the next key
    kv: Dict[str, str] = {}
    key_iter = list(re.finditer(r"(?P<key>[a-zA-Z_]+):", line))
    if not key_iter:
        return kv
    for i, m in enumerate(key_iter):
        key = m.group("key")
        val_start = m.end()
        val_end = key_iter[i + 1].start() if i + 1 < len(key_iter) else len(line)
        kv[key] = line[val_start:val_end].strip()
    return kv


def _parse_aishell_line(line: str) -> Optional[AishellSegment]:
    """
    Parse one line in AIShell manifest, e.g.
      utt:BAC009S0104W0446 feat:/abs/path/to/cut.wav feat_shape:DUR text:....

    IMPORTANT:
    - Each line should correspond to exactly one pre-cut wav file.
    """
    line = line.strip()
    if not line:
        return None

    kv = _parse_kv_line(line)

    if "utt" not in kv or "feat" not in kv or "text" not in kv or "feat_shape" not in kv:
        return None

    wav_path = kv["feat"].strip()
    if not wav_path:
        return None

    try:
        duration_sec = float(kv["feat_shape"])
    except Exception:
        return None
    if duration_sec <= 0:
        return None

    return AishellSegment(
        utt_id=kv["utt"],
        wav_path=wav_path,
        duration_sec=duration_sec,
        text=kv["text"],
    )


def aishell_text_to_char_tokens(text: str) -> str:
    """
    Convert raw transcript into character-level tokens:
    - inter-character separator: " "
    - inter-word separator: "|"
    - append a trailing " |"
    """
    text = (text or "").strip()
    text = text.upper()
    # Normalize unknown marker if any
    text = text.replace("<UNK>", "").replace("<unk>", "")
    # Word boundary marker
    text = text.replace(" ", "|")
    return " ".join(list(text)) + " |"


def load_aishell_manifests(manifest_paths: Sequence[str]) -> List[AishellSegment]:
    segments: List[AishellSegment] = []
    for mp in manifest_paths:
        with open(mp, "r", encoding="utf-8") as f:
            for line in f:
                seg = _parse_aishell_line(line)
                if seg is not None:
                    segments.append(seg)
    return segments


class SequenceDataset(Dataset):
    """
    AIShell ASR dataset:
    - Input manifests are *.data.cut files containing (cut_wav_path, duration, text)
    - Pre-bucket segments by duration for efficiency
    """

    def __init__(
        self,
        split: str,
        bucket_size: int,
        dictionary,
        manifest_dir: str,
        **kwargs,
    ):
        super().__init__()
        self.dictionary = dictionary
        self.sample_rate = SAMPLE_RATE

        manifest_names = kwargs[split]
        assert isinstance(manifest_names, (list, tuple)) and len(manifest_names) > 0, (
            f"Expected config.downstream_expert.datarc['{split}'] to be a non-empty list of manifest filenames"
        )
        manifest_paths = [
            m if os.path.isabs(m) else os.path.join(manifest_dir, m) for m in manifest_names
        ]
        for p in manifest_paths:
            assert os.path.isfile(p), f"Manifest not found: {p}"

        segments = load_aishell_manifests(manifest_paths)
        assert len(segments) > 0, f"0 data found for {split}"

        # Tokenize + encode labels
        self._segments: Dict[str, AishellSegment] = {s.utt_id: s for s in segments}
        self.Y = {
            utt_id: self.dictionary.encode_line(
                aishell_text_to_char_tokens(seg.text),
                line_tokenizer=lambda x: x.split(),
                add_if_not_exist=False,
            ).long()
            for utt_id, seg in self._segments.items()
        }

        # Sort by duration (desc) for bucketing
        segments_sorted = sorted(segments, key=lambda s: s.duration_sec, reverse=True)

        self.X: List[List[str]] = []
        batch_ids: List[str] = []
        batch_durs: List[float] = []
        for seg in tqdm(
            segments_sorted,
            total=len(segments_sorted),
            desc=f"AIShell dataset {split}",
            dynamic_ncols=True,
        ):
            batch_ids.append(seg.utt_id)
            batch_durs.append(seg.duration_sec)
            if len(batch_ids) == bucket_size:
                if bucket_size >= 2 and max(batch_durs) > HALF_BATCHSIZE_TIME_SEC:
                    half = bucket_size // 2
                    self.X.append(batch_ids[:half])
                    self.X.append(batch_ids[half:])
                else:
                    self.X.append(batch_ids)
                batch_ids, batch_durs = [], []

        if len(batch_ids) > 0:
            self.X.append(batch_ids)

    def __len__(self):
        return len(self.X)

    def _load_wav(self, seg: AishellSegment, utt_id: str):
        """
        Load a pre-cut wav (full file) as 1-D float waveform.
        """
        def _raise_with_context(e: Exception):
            raise RuntimeError(
                "Failed to decode AIShell cut wav.\n"
                f"  utt_id={utt_id}\n"
                f"  wav_path={seg.wav_path}\n"
                f"  duration_sec={seg.duration_sec}\n"
                f"  original_error={repr(e)}"
            ) from e

        # 1) torchaudio load (fast path)
        try:
            wav, sr = torchaudio.load(seg.wav_path)
            assert sr == self.sample_rate, (
                f"Sample rate mismatch: real {sr}, expected {self.sample_rate} ({seg.wav_path})"
            )
            return wav.view(-1)
        except Exception:
            pass

        # 2) stdlib wave (PCM WAV only) fallback
        try:
            with wave.open(seg.wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                if sr != self.sample_rate:
                    raise RuntimeError(
                        f"Sample rate mismatch: real {sr}, expected {self.sample_rate}"
                    )
                if sampwidth != 2:
                    raise RuntimeError(
                        f"Unsupported sampwidth={sampwidth} (expected 16-bit PCM)"
                    )
                raw = wf.readframes(wf.getnframes())

            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if data.size == 0:
                data = np.zeros((1,), dtype=np.float32)
            elif n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            return torch.from_numpy(data).view(-1)
        except Exception as e:
            _raise_with_context(e)

    def __getitem__(self, index):
        utt_ids = self.X[index]
        wav_batch = [
            self._load_wav(self._segments[uid], uid).numpy() for uid in utt_ids
        ]
        label_batch = [self.Y[uid].numpy() for uid in utt_ids]
        filename_batch = utt_ids
        return wav_batch, label_batch, filename_batch

    def collate_fn(self, items):
        # bucketing: dataset already returns a bucket, DataLoader batch_size=1
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2]

