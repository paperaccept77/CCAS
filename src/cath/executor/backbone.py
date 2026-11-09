"""Local backbone and role-specific adapter construction."""
import torch
from ..types import Role


def load_backbone(config):
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(config.backbone, local_files_only=True)
    base = AutoModel.from_pretrained(config.backbone, local_files_only=True,
                                     torch_dtype=getattr(torch, config.precision))
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none")
    backbone = get_peft_model(base, lora, adapter_name=Role(0).name.lower())
    for role in list(Role)[1:]:
        backbone.add_adapter(role.name.lower(), lora)
    return tokenizer, backbone, base.config.hidden_size
