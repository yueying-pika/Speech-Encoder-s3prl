import inspect
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoderLayer
import inspect
print("attn_impl:", getattr(self.config, "_attn_implementation", None))
print("self_attn class:", type(self.self_attn))
print("self_attn module:", self.self_attn.__class__.__module__)
print("self_attn file:", inspect.getsourcefile(self.self_attn.__class__))
print("self_attn forward sig:", inspect.signature(self.self_attn.forward))