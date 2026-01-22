#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Omni Upstream Entry Points for S3PRL
"""

# 每次调用qwen推理, expert.py
# 使预提取特征推理, expert_cached.py
from .expert import UpstreamExpert as _UpstreamExpert
from .expert_cached import UpstreamExpert as _CachedUpstreamExpert
from .expert_online import UpstreamExpert as _OnlineUpstreamExpert

def qwen_omni_local(ckpt: str, *args, **kwargs):
    """从本地路径加载 Qwen2.5-Omni(在线推理)"""
    return _UpstreamExpert(ckpt=ckpt, **kwargs)


def qwen_omni_custom(ckpt: str, *args, **kwargs):
    """从自定义路径加载"""
    return _UpstreamExpert(ckpt=ckpt, **kwargs)


def qwen_omni(refresh: bool = False, *args, **kwargs):
    """从 HuggingFace 加载"""
    return _UpstreamExpert(ckpt="Qwen/Qwen2.5-Omni-7B", **kwargs)


def qwen_omni_7b(*args, **kwargs):
    """预设路径的 Qwen2.5-Omni-7B(在线推理)"""
    default_path = "/home/s2025244189/s1025244003/Qwen2.5-Omni-7B"
    kwargs.pop('ckpt', None)
    return _UpstreamExpert(ckpt=default_path, **kwargs)

def qwen_omni_online(*args, **kwargs):
    """预设路径的 Qwen2.5-Omni-7B """
    default_path = "/home/s2025244189/s1025244003/Qwen2.5-Omni-7B"
    kwargs.pop('ckpt', None)
    return _OnlineUpstreamExpert(ckpt=default_path, **kwargs)

def qwen_omni_cached(*args, **kwargs):
    """ 从预提取的特征加载"""
    default_path = "/home/s2025244189/s1025244003/s3prl/s3prl/data/hidden_layer_all"
    kwargs.pop('ckpt', None)
    return _CachedUpstreamExpert(ckpt=default_path, **kwargs)
