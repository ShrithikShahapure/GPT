import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

def main(ckpt_path: str, out_dir: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"]
    cfg = ckpt["config"]  

    hf_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,        
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    model = GPT2LMHeadModel(hf_cfg)
    sd_hf = model.state_dict()

    
    for k in sd_hf.keys():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue
        if any(k.endswith(suf) for suf in TRANSPOSED_SUFFIXES):
            sd_hf[k].copy_(sd[k].t())
        else:
            sd_hf[k].copy_(sd[k])

    model.load_state_dict(sd_hf, strict=False)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.ckpt, args.out)
