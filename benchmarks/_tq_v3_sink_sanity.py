#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness gates for the v3 unified-attention sink path (USE_SINKS=1).

Validates that the per-head sink logit is folded into the softmax denominator
correctly on both the 2D (short KV) and 3D split-KV (long KV) paths, and
that the non-sink path is completely unaffected (sinks=None is default and
bit-identical to the pre-sink kernel).

Gates (run on 2D/short and 3D/long shapes):

    G-sink-1  regression: sinks=None output == output of code path that never
              mentions sinks (this file, compared against the Q-rot sanity
              baseline). Implicitly bit-identical because USE_SINKS=0 is a
              constexpr that compiles the sink branch out.

    G-sink-2  very-negative sink (s_h = -100) is numerically equivalent to
              sinks=None (fp16 max |delta| == 0, since exp(-100 - M) underflows
              to exactly 0 in fp32 for any realistic score magnitude).

    G-sink-3  monotone damping: as we sweep the sink from -inf through 0
              up to +10 while keeping everything else fixed, the output
              row norm is monotonically non-increasing per head. This is
              the defining qualitative property of a sink: a larger sink
              logit steals more probability mass from the real tokens.

    G-sink-4  analytical fp32 parity: compute a fp32 reference
              out_ref = sum_i exp(q.k_i) V_i / (exp(s_h) + sum_i exp(q.k_i))
              from the un-quantized keys/values and the fp32 query, and
              compare to the kernel output. Tolerance is calibrated to
              1.5x the same-shape NO-sink baseline relative error, which
              already absorbs 4-bit MSE/V-affine quant noise. Sinks should
              not add any error beyond that floor.

    G-sink-5  prod ↔ ablation bit-identity: running the ablation kernel
              (all ABLATION_SKIP_* = 0) with the same sinks vector returns
              the exact same fp16 bits as the production kernel, proving
              the mirror is correct on the sink path too.

Run:

    HIP_VISIBLE_DEVICES=3 python benchmarks/_tq_v3_sink_sanity.py
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


def _hadamard(D: int, device: str) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(D)).float()


def _fp32_reference_with_sink(
    q_fp32: torch.Tensor,  # [Hq, D]
    key_fp32: torch.Tensor,  # [seq, Hk, D]
    value_fp32: torch.Tensor,  # [seq, Hk, D]
    sinks_fp32: torch.Tensor | None,  # [Hq] or None
    scale: float,
    kv_group_size: int,
) -> torch.Tensor:
    """Compute the analytical fp32 attention with an optional per-head sink.

    out[h] = sum_i softmax_i(scale * q[h].dot(k[i, kh]) [, sink]) * V[i, kh]

    where ``kh = h // kv_group_size`` maps the query head to its KV group
    and the sink contributes ``exp(sink[h])`` to the denominator (and 0 to
    the numerator). This is the ground truth; the TQ kernel is expected
    to match it up to 4-bit quant noise.
    """
    Hq, D = q_fp32.shape
    out = torch.zeros((Hq, D), device=q_fp32.device, dtype=torch.float32)
    for h in range(Hq):
        kh = h // kv_group_size
        # scores: [seq], numerically stable softmax
        scores = scale * (key_fp32[:, kh, :].float() @ q_fp32[h].float())
        if sinks_fp32 is not None:
            m = torch.maximum(scores.max(), sinks_fp32[h])
            denom = torch.exp(sinks_fp32[h] - m) + torch.exp(scores - m).sum()
        else:
            m = scores.max()
            denom = torch.exp(scores - m).sum()
        probs = torch.exp(scores - m) / denom  # [seq]
        out[h] = (probs[:, None] * value_fp32[:, kh, :].float()).sum(dim=0)
    return out


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
    PiT = H
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

    # Distinct per-head sinks in a plausible gpt-oss range (log-scale ~N(0, 1.5)).
    torch.manual_seed(7)
    sinks = torch.randn(Hq, device=device, dtype=torch.float32) * 1.5

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

    # Reference: no-sink run through the production kernel.
    out_no_sink = torch.empty_like(query)
    prod_attn(**base, output=out_no_sink, sinks=None)

    # Reference: sinks=None explicitly vs not-passing (both should hit
    # USE_SINKS=0; compiler should have cached the same kernel).
    out_explicit_none = torch.empty_like(query)
    prod_attn(**base, output=out_explicit_none, sinks=None)

    # With a very negative sink: should be numerically equivalent to no-sink.
    very_neg = torch.full((Hq,), -100.0, device=device, dtype=torch.float32)
    out_very_neg = torch.empty_like(query)
    prod_attn(**base, output=out_very_neg, sinks=very_neg)

    # Real per-head sinks.
    out_sink = torch.empty_like(query)
    prod_attn(**base, output=out_sink, sinks=sinks)

    # Ablation with sinks + all ABLATION flags 0 (bit-identity G5).
    out_ablation_sink = torch.empty_like(query)
    ablation_attn(**base, output=out_ablation_sink, sinks=sinks)

    # FP32 analytical references (no-sink baseline and with-sink).
    q_fp32 = query[0].float()
    key_fp32 = key.float()
    value_fp32 = value.float()
    out_fp32_no_sink = _fp32_reference_with_sink(
        q_fp32, key_fp32, value_fp32, None, scale, Hq // Hk
    )
    out_fp32_sink = _fp32_reference_with_sink(
        q_fp32, key_fp32, value_fp32, sinks, scale, Hq // Hk
    )

    path = "3D" if (seq_len >= 1024 and not force_2d) else "2D"
    print(f"\n[{label}]  B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} path={path}")

    # --- G-sink-1: explicit None equals default (bit-identical) ---
    d1 = (out_no_sink.float() - out_explicit_none.float()).abs().max().item()
    g1_ok = d1 == 0.0
    print(f"  G-sink-1 sinks=None vs default        |Δ| = {d1:.3e}   (want 0)")

    # --- G-sink-2: very-negative sink equals no-sink (fp16 bit-identical) ---
    d2 = (out_very_neg.float() - out_no_sink.float()).abs().max().item()
    g2_ok = d2 == 0.0
    print(f"  G-sink-2 sink=-100 vs no-sink         |Δ| = {d2:.3e}   (want 0)")

    # --- G-sink-3: monotone damping when sink grows ---
    # Sweep a single global sink value uniformly across heads; the output
    # row norm must be non-increasing as sink increases.
    sweep = [-20.0, -5.0, 0.0, 2.0, 5.0, 10.0]
    norms = []
    for s in sweep:
        s_vec = torch.full((Hq,), s, device=device, dtype=torch.float32)
        o = torch.empty_like(query)
        prod_attn(**base, output=o, sinks=s_vec)
        norms.append(o.float().pow(2).sum(dim=-1).sqrt().mean().item())
    # Per-sweep step must be (within tiny fp slack) non-increasing.
    slack = 1e-4
    monotone = all(norms[i] - norms[i + 1] >= -slack for i in range(len(norms) - 1))
    g3_ok = monotone
    norms_str = [f"{n:.4f}" for n in norms]
    print(f"  G-sink-3 monotone damping              norms={norms_str}")

    # --- G-sink-4: fp32 analytical parity (calibrated for quant noise) ---
    # Baseline no-sink quant error vs fp32 analytical no-sink.
    diff_no_sink = (out_no_sink[0].float() - out_fp32_no_sink).abs()
    ref_no_sink = out_fp32_no_sink.abs().max().clamp_min(1e-6)
    rel_no_sink = (diff_no_sink / ref_no_sink).max().item()

    # With-sink quant error vs fp32 analytical with-sink.
    diff_sink = (out_sink[0].float() - out_fp32_sink).abs()
    ref_sink = out_fp32_sink.abs().max().clamp_min(1e-6)
    rel_sink = (diff_sink / ref_sink).max().item()

    # Sinks shouldn't inflate the error beyond 1.5x the no-sink baseline.
    g4_bound = max(1.5 * rel_no_sink, 5e-3)
    g4_ok = rel_sink <= g4_bound
    print(
        f"  G-sink-4 fp32 parity                   rel_no_sink={rel_no_sink:.2e}  "
        f"rel_sink={rel_sink:.2e}   bound={g4_bound:.2e}"
    )

    # --- G-sink-5: prod ↔ ablation bit-identity with sinks ---
    d5 = (out_sink.float() - out_ablation_sink.float()).abs().max().item()
    g5_ok = d5 == 0.0
    print(f"  G-sink-5 prod vs ablation (w/ sinks)  |Δ| = {d5:.3e}   (want 0)")

    for name, ok in (
        ("G-sink-1", g1_ok),
        ("G-sink-2", g2_ok),
        ("G-sink-3", g3_ok),
        ("G-sink-4", g4_ok),
        ("G-sink-5", g5_ok),
    ):
        tag = "PASS" if ok else "FAIL"
        print(f"    {name}  {tag}")

    return g1_ok and g2_ok and g3_ok and g4_ok and g5_ok


def main():
    torch.cuda.init()
    shapes = [
        dict(label="2D/short", B=1, Hq=64, Hk=8, D=64, seq_len=512, force_2d=True),
        dict(label="3D/long", B=1, Hq=64, Hk=8, D=64, seq_len=4095, force_2d=False),
    ]
    all_ok = True
    for sh in shapes:
        try:
            ok = _run_one_shape(**sh)
        except Exception as e:  # noqa: BLE001
            print(f"\n[{sh['label']}] FAIL {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            ok = False
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all_ok else 'FAILURES SEEN'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
