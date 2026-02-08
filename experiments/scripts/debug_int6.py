#!/usr/bin/env python3
"""Debug INT6 anomaly on 7B BF16"""
import os, torch
os.environ['TRANSFORMERS_NO_TF'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True, attn_implementation='eager')
model.eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

prompt = "Context: Super Bowl 50 was won by the Denver Broncos who defeated Carolina Panthers 24-10.\nQuestion: Which team won Super Bowl 50?\nAnswer:"
inputs = tok(prompt, return_tensors="pt").to("cuda")
sl = inputs['input_ids'].shape[1]

def quantize_tensor(t, bits):
    if bits >= 16: return t
    qmin = -(2**(bits-1))
    qmax = 2**(bits-1)-1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    t_q = (t / scale).round().clamp(qmin, qmax)
    return t_q * scale

# Test generation with each quantization level
for bits in [4, 5, 6, 7, 8]:
    with torch.no_grad():
        out = model(input_ids=inputs['input_ids'], use_cache=True)
    pkv = out.past_key_values

    # Quantize
    for li in range(len(pkv.layers)):
        l = pkv.layers[li]
        l.keys.copy_(quantize_tensor(l.keys, bits))
        l.values.copy_(quantize_tensor(l.values, bits))

    first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
    gen = [first_tok]
    cur = sl
    mask = torch.ones(1, sl, device='cuda', dtype=torch.long)
    for step in range(30):
        ni = torch.tensor([[gen[-1]]], device='cuda')
        pi = torch.tensor([[cur]], device='cuda')
        mask = torch.cat([mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)
        with torch.no_grad():
            o = model(input_ids=ni, past_key_values=pkv, attention_mask=mask, position_ids=pi, use_cache=True)
        pkv = o.past_key_values
        nt = o.logits[:, -1, :].argmax(dim=-1).item()
        gen.append(nt)
        cur += 1
        if nt == tok.eos_token_id: break
    ans = tok.decode(gen, skip_special_tokens=True).strip()[:100]
    print(f"INT{bits}: {ans}")

# Also check error magnitudes per layer for INT6
print("\n--- INT6 error per layer (first 5 layers) ---")
with torch.no_grad():
    out = model(input_ids=inputs['input_ids'], use_cache=True)
pkv = out.past_key_values
for li in range(min(5, len(pkv.layers))):
    l = pkv.layers[li]
    k_orig = l.keys.clone()
    v_orig = l.values.clone()
    k_q = quantize_tensor(k_orig, 6)
    v_q = quantize_tensor(v_orig, 6)
    k_err = (k_orig - k_q).abs().mean().item()
    v_err = (v_orig - v_q).abs().mean().item()
    k_range = k_orig.abs().max().item()
    v_range = v_orig.abs().max().item()
    print(f"  Layer {li}: key_err={k_err:.6f} (range={k_range:.3f}) | val_err={v_err:.6f} (range={v_range:.3f})")
