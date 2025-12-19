from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# -----------------------------------------------------------------------------------------------------------------

class CasualSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        

    def forward(self,x):

        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd,dim =2)
        
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd )
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head:     int = 12
    n_layer:    int = 12
    n_embd:     int = 768


class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer= nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size,config.n_embd),
            wpe  = nn.Embedding(config.vocab_size,config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),

        ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias = False)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
                std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



    @classmethod
    def from_pretrained(cls,model_type):

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print("Loading weights from GPT: %s",model_type)

        config_args = {
            'gpt2':            dict(n_layer = 12,n_head = 12, n_embd = 768),
            'gpt2-medium':     dict(n_layer = 24,n_head =16, n_embd = 1024),
            'gpt2-large':      dict(n_layer = 36,n_head = 20, n_embd = 1280),
            'gpt2-xl':         dict(n_layer = 48,n_head = 25, n_embd = 1600),

        }[model_type]

        config_args['vocab_size']   = 50257
        config_args['block_size']   = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]


        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):

                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
            
# -------------------------------------------------------------------------------------------------------------------
# logs

# /Users/shrithkshahapure/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
#   warnings.warn(
# Loading weights from GPT: %s gpt2
# Traceback (most recent call last):
#   File "/Users/shrithkshahapure/repos/ml-training/GPT/train-model.py", line 355, in <module>
#     model = GPT.from_pretrained('gpt2')
#   File "/Users/shrithkshahapure/repos/ml-training/GPT/train-model.py", line 172, in from_pretrained
#     assert sd_hf[k].shape == sd[k].shape
# AssertionError


# -------------------------------------------------------------------------------------------------------------------

model = GPT.from_pretrained('gpt2')
print("Did not crash lmao")