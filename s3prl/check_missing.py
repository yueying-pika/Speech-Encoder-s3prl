#!/usr/bin/env python3
from pathlib import Path

libri_root = Path("/home/s2025244189/datas/LibriSpeech")
feat_root = Path("/home/s2025244189/s1025244003/s3prl/s3prl/data/hidden_new")

splits = ['train-clean-100', 'train-clean-360', 'train-other-500']
split_map = {'train-clean-100': 'train_clean_100', 
             'train-clean-360': 'train_clean_360',
             'train-other-500': 'train_other_500'}

for split in splits:
    audio_dir = libri_root / split
    feat_dir = feat_root / f"lib_hidden_{split_map[split]}"
    
    missing = []
    for flac in audio_dir.rglob("*.flac"):
        rel = flac.relative_to(audio_dir)
        feat_path = feat_dir / rel.parent / f"{flac.stem}_seq.npz"
        if not feat_path.exists():
            missing.append(str(flac))
    
    print(f"{split}: {len(missing)} missing files")
    
    # 保存缺失文件列表
    if missing:
        with open(f"missing_{split_map[split]}.txt", "w") as f:
            f.write("\n".join(missing))
        print(f"  Saved to: missing_{split_map[split]}.txt")

