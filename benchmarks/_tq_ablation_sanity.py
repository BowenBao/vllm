#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick sanity for the rebased ablation kernel.

(1) flags-all-0 → bit-identical to production
(2) each skip_* flag compiles and runs

Run: HIP_VISIBLE_DEVICES=3 python benchmarks/_tq_ablation_sanity.py
"""

from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, ".")

from vllm.model_executor.layers.quantization.turboquant.centroids import solve_lloyd_max
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_unified_attention as prod_attn,
)
from vllm.v1.attention.ops.triton_turboquant_unified_attention_ablation import (
    triton_turboquant_unified_attention as ablation_attn,
)


def _hadamard(D, device):
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(D)).float()


def main():
    device = "cuda"
    B, Hq, Hk, D = 1, 64, 8, 64
    seq_len = 2047
    block_size = 16
    preset = "turboquant_4bit_nc"

    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
    H = _hadamard(D, device)
    PiT = H
    centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    torch.manual_seed(42)
    key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    num_blocks = (seq_len + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        Hk,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    triton_turboquant_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        PiT,
        midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        centroids=centroids,
        norm_correction=cfg.norm_correction,
    )

    torch.manual_seed(99)
    query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
    query_start_loc = torch.arange(B + 1, device=device, dtype=torch.int32)
    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(
        0
    )
    scale = 1.0 / math.sqrt(D)

    base = dict(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        Pi=PiT,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        value_packed_size=cfg.value_packed_size,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
        max_query_len=1,
        max_seq_len=seq_len,
        force_2d=True,
    )

    out_prod = torch.empty_like(query)
    out_ablation = torch.empty_like(query)
    prod_attn(**base, output=out_prod)
    ablation_attn(**base, output=out_ablation)

    diff = (out_prod.float() - out_ablation.float()).abs().max().item()
    print(f"[bit-identity] max|prod - ablation(flags=0)| = {diff:.3e}")
    ok_ident = diff == 0.0
    print(f"[bit-identity] {'PASS' if ok_ident else 'FAIL (nonzero)'}")

    variants = [
        "ablation_skip_k",
        "ablation_skip_centroid",
        "ablation_skip_norm_load",
        "ablation_skip_v",
        "ablation_skip_v_scale_load",
    ]
    print()
    print("Per-variant compile+run (value check skipped; only crash check):")
    fails = []
    for name in variants:
        try:
            o = torch.empty_like(query)
            ablation_attn(**base, output=o, **{name: 1})
            mx = o.float().abs().max().item()
            mean = o.float().abs().mean().item()
            print(f"  {name:>22}=1 : max|o|={mx:.4e}  mean|o|={mean:.4e}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:>22}=1 : FAIL {type(e).__name__}: {e}")
            fails.append(name)

    # Also try both K and V skipped simultaneously — must still run.
    try:
        o = torch.empty_like(query)
        ablation_attn(**base, output=o, ablation_skip_k=1, ablation_skip_v=1)
        mx = o.float().abs().max().item()
        print(f"  {'skip_k+skip_v':>22}   : max|o|={mx:.4e}  (expect 0.0)")
    except Exception as e:  # noqa: BLE001
        print(f"  skip_k+skip_v FAIL {type(e).__name__}: {e}")
        fails.append("skip_k+skip_v")

    print()
    print(
        f"Summary: identity={'PASS' if ok_ident else 'FAIL'} "
        f"variants_failed={len(fails)}"
    )
    sys.exit(0 if ok_ident and not fails else 1)


if __name__ == "__main__":
    main()
