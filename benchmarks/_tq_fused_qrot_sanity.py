#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness gates for the fused Q-rotation prologue (Opt-FuseQRot).

Tests three properties on a realistic decode shape against the current
production kernel (which now has fuse_q_rot=True as the default):

    G1  numerical parity        : |fused - legacy_launcher_rot| is small
                                  (max relative delta < 5e-4 on fp16 outputs)
    G2  ulp distribution        : 99th %ile ulp delta <= 4
                                  (fp16 bit-level diff is tightly bounded)
    G3  bit-identity fallback   : calling prod_attn(fuse_q_rot=False) is
                                  bit-for-bit identical to ablation kernel
                                  with fuse_q_rot=False + all ablation flags 0
                                  (proves the legacy path is untouched)

Both 2D (short KV, force_2d=True) and 3D split-KV (long KV) paths are
exercised. Run:

    HIP_VISIBLE_DEVICES=3 python benchmarks/_tq_fused_qrot_sanity.py
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

# --------------------------------------------------------------------------- #
# Gate thresholds — consciously looser than v1↔v3 because we're comparing a
# Triton MFMA against rocBLAS fp32 GEMM. Rounding-order differs; algebra is
# identical.
# --------------------------------------------------------------------------- #
MAX_REL_DELTA = 5e-4  # max |fused - legacy| / max |legacy|
ULP99_MAX = 4  # 99th %ile fp16 ulp delta


def _hadamard(D: int, device: str) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(D)).float()


def _fp16_ulp_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return |int(a_fp16_bits) - int(b_fp16_bits)| element-wise.

    Works for same-signed magnitude comparisons around zero (which is how
    IEEE-754 lays out fp16 so ulp distance equals |int diff| on the raw
    bit pattern). We use this only as an order-of-magnitude bound.
    """
    a16 = a.to(torch.float16).contiguous()
    b16 = b.to(torch.float16).contiguous()
    ai = a16.view(torch.int16).to(torch.int32)
    bi = b16.view(torch.int16).to(torch.int32)

    # Convert from sign-magnitude to biased order so the subtraction is
    # monotonic across 0 as well.
    def _biased(x):
        neg = x < 0
        x = torch.where(neg, (-x) | 0x8000, x)
        return x

    return (_biased(ai) - _biased(bi)).abs()


def _percentile(t: torch.Tensor, p: float) -> float:
    flat = t.flatten().to(torch.float32)
    k = int(flat.numel() * p)
    k = max(0, min(flat.numel() - 1, k))
    return float(flat.kthvalue(k + 1).values)


def _run_one_shape(
    *,
    label: str,
    B: int,
    Hq: int,
    Hk: int,
    D: int,
    seq_len: int,
    force_2d: bool,
) -> bool:
    device = "cuda"
    preset = "turboquant_4bit_nc"
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)

    H = _hadamard(D, device)
    PiT = H  # orthonormal, so PiT = Pi.T up to a sign convention; kernel
    # treats it opaquely. Matches the driver convention.
    centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    torch.manual_seed(42)
    key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    block_size = 16
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
        force_2d=force_2d,
    )

    # ---- legacy launcher rotation (fuse_q_rot=False) ----
    out_legacy = torch.empty_like(query)
    prod_attn(**base, output=out_legacy, fuse_q_rot=False)

    # ---- fused in-kernel rotation (fuse_q_rot=True, the new default) ----
    out_fused = torch.empty_like(query)
    prod_attn(**base, output=out_fused, fuse_q_rot=True)

    # ---- ablation kernel with fuse off + all ablation flags 0 (for G3) ----
    out_ablation_legacy = torch.empty_like(query)
    ablation_attn(**base, output=out_ablation_legacy, fuse_q_rot=False)

    # G1: numerical parity fused vs legacy launcher rot.
    diff = (out_fused.float() - out_legacy.float()).abs()
    ref_mag = out_legacy.float().abs().max().item()
    max_rel = diff.max().item() / max(ref_mag, 1e-6)

    # G2: fp16 ulp distribution.
    ulp = _fp16_ulp_delta(out_fused, out_legacy)
    ulp_max = int(ulp.max().item())
    ulp_p99 = int(_percentile(ulp, 0.99))
    ulp_p999 = int(_percentile(ulp, 0.999))

    # G3: bit-identity of legacy paths (prod fuse=False vs ablation fuse=False flags=0)
    bit_diff = (out_legacy.float() - out_ablation_legacy.float()).abs().max().item()

    path = "3D" if (seq_len >= 1024 and not force_2d) else "2D"
    print(f"\n[{label}]  B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} path={path}")
    print(
        f"  G1 max|fused - legacy|   = {diff.max().item():.3e}"
        f"   max |legacy| = {ref_mag:.3e}"
        f"   max_rel = {max_rel:.2e}   "
        f"threshold = {MAX_REL_DELTA:.1e}"
    )
    print(
        f"  G2 fp16 ulp delta        max={ulp_max:<3d}  "
        f"p99={ulp_p99:<3d}  p999={ulp_p999:<3d}   "
        f"threshold p99 <= {ULP99_MAX}"
    )
    print(
        f"  G3 bit-identity legacy   max|prod(fuse=F) - ablation(fuse=F,flags=0)| "
        f"= {bit_diff:.3e}"
    )

    g1_ok = max_rel <= MAX_REL_DELTA
    g2_ok = ulp_p99 <= ULP99_MAX
    g3_ok = bit_diff == 0.0
    for name, ok in (("G1", g1_ok), ("G2", g2_ok), ("G3", g3_ok)):
        tag = "PASS" if ok else "FAIL"
        print(f"  {name}  {tag}")
    return g1_ok and g2_ok and g3_ok


def main():
    torch.cuda.init()
    shapes = [
        # Short KV, force 2D path.
        dict(label="2D/short", B=1, Hq=64, Hk=8, D=64, seq_len=512, force_2d=True),
        # Long KV, 3D split-KV path (seq_len >= 1024 + not force_2d).
        dict(label="3D/long", B=1, Hq=64, Hk=8, D=64, seq_len=2047, force_2d=False),
        # Even longer KV to exercise multi-segment path more aggressively.
        dict(label="3D/xl", B=1, Hq=64, Hk=8, D=64, seq_len=8191, force_2d=False),
    ]
    all_ok = True
    for sh in shapes:
        try:
            ok = _run_one_shape(**sh)
        except Exception as e:  # noqa: BLE001
            print(f"\n[{sh['label']}] FAIL {type(e).__name__}: {e}")
            ok = False
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all_ok else 'FAILURES SEEN'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
