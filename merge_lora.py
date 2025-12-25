import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_dir = "hf_pretrained"
lora_dir = "hf_sft_lora"
out_dir  = "hf_sft_merged"

tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, lora_dir)
model = model.merge_and_unload()

model.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)
