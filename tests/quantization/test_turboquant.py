# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant KV-cache quantization.

Run: .venv/bin/python -m pytest tests/quantization/test_turboquant.py -v
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

# ============================================================================
# Helpers
# ============================================================================

ALL_PRESETS = list(TQ_PRESETS.keys())


def _assert_strictly_sorted(seq, name="sequence"):
    for i in range(len(seq) - 1):
        assert seq[i] < seq[i + 1], f"{name} not sorted at index {i}"


def _is_power_of_2(n: int) -> bool:
    return n > 0 and next_power_of_2(n) == n


# Expected concrete values for each preset at head_dim=128.
# fmt: off
PRESET_EXPECTED = {
    "turboquant_k8v4": dict(
        key_fp8=True,  key_quant_bits=8,
        key_mse_bits=0, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=False,
        key_packed_size=128, value_packed_size=68,
        slot_size=196, slot_size_aligned=196,
    ),
    "turboquant_4bit_nc": dict(
        key_fp8=False, key_quant_bits=4,
        key_mse_bits=4, value_quant_bits=4,
        mse_bits=4, n_centroids=16, centroid_bits=4,
        norm_correction=True,
        key_packed_size=66, value_packed_size=68,
        slot_size=134, slot_size_aligned=134,
    ),
    "turboquant_k3v4_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=4,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=68,
        slot_size=118, slot_size_aligned=118,
    ),
    "turboquant_3bit_nc": dict(
        key_fp8=False, key_quant_bits=3,
        key_mse_bits=3, value_quant_bits=3,
        mse_bits=3, n_centroids=8, centroid_bits=3,
        norm_correction=True,
        key_packed_size=50, value_packed_size=52,
        slot_size=102, slot_size_aligned=102,
    ),
}
# fmt: on


# ============================================================================
# Config tests (CPU-only, no dependencies beyond config.py)
# ============================================================================


class TestTurboQuantConfig:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_parses(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert isinstance(cfg, TurboQuantConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("turboquant_invalid", head_dim=128)

    # ---- Per-preset concrete value checks (table-driven) ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_fp8 is exp["key_fp8"]
        assert cfg.key_quant_bits == exp["key_quant_bits"]
        assert cfg.key_mse_bits == exp["key_mse_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_value_mode(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.value_quant_bits == exp["value_quant_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_bits_and_centroids(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.mse_bits == exp["mse_bits"]
        assert cfg.n_centroids == exp["n_centroids"]
        assert cfg.centroid_bits == exp["centroid_bits"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_norm_correction(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.norm_correction is PRESET_EXPECTED[preset]["norm_correction"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_packed_sizes(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        exp = PRESET_EXPECTED[preset]
        assert cfg.key_packed_size == exp["key_packed_size"]
        assert cfg.value_packed_size == exp["value_packed_size"]
        assert cfg.slot_size == exp["slot_size"]
        assert cfg.slot_size_aligned == exp["slot_size_aligned"]

    # ---- Cross-preset structural invariants ----

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_slot_equals_key_plus_value(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_padded_slot_is_even(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0, (
            f"slot_size_aligned={cfg.slot_size_aligned} is not even"
        )

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_key_value_packed_sizes_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.key_packed_size > 0
        assert cfg.value_packed_size > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_n_centroids_is_2_to_mse_bits(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.n_centroids == 2**cfg.mse_bits

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_centroid_bits_always_positive(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.centroid_bits > 0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_mse_key_or_fp8_exclusive(self, preset):
        """Each preset is either FP8 keys or MSE keys, never both."""
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        if cfg.key_fp8:
            assert cfg.key_mse_bits == 0
            assert cfg.key_quant_bits == 8
        else:
            assert cfg.key_mse_bits > 0
            assert cfg.key_quant_bits in (3, 4)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_all_presets_all_head_dims(self, preset, head_dim):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)
        assert cfg.head_dim == head_dim
        assert cfg.slot_size == cfg.key_packed_size + cfg.value_packed_size
        assert cfg.slot_size_aligned >= cfg.slot_size
        assert cfg.slot_size_aligned % 2 == 0

    # ---- Boundary skip layers ----

    @staticmethod
    def _dense_model_config(num_layers):
        from types import SimpleNamespace

        return SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(num_hidden_layers=num_layers),
        )

    def test_boundary_skip_layers_basic(self):
        mc = self._dense_model_config(32)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_layers_zero(self):
        mc = self._dense_model_config(32)
        assert TurboQuantConfig.get_boundary_skip_layers(mc, 0) == []

    def test_boundary_skip_layers_small_model(self):
        mc = self._dense_model_config(4)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc)
        assert layers == ["0", "1", "2", "3"]

    def test_boundary_skip_layers_cap_at_half(self):
        mc = self._dense_model_config(8)
        layers = TurboQuantConfig.get_boundary_skip_layers(mc, 10)
        assert len(layers) == 8


class TestHybridAttentionIndices:
    """Regression tests for boundary protection on hybrid models.

    Hybrid models (attention + Mamba / linear-attention) identify KV-carrying
    layers via layer_types / layers_block_type / attn_type_list. The helper
    must return the *global* layer indices of the full-attention layers so
    that kv_cache_dtype_skip_layers matches what extract_layer_index(prefix)
    reports on the Attention layers at runtime.
    """

    @staticmethod
    def _fake_model_config(text_cfg=None, hf_cfg=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_text_config=text_cfg if text_cfg is not None else SimpleNamespace(),
            hf_config=hf_cfg if hf_cfg is not None else SimpleNamespace(),
        )

    def test_layer_types_full_attention(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layer_types = [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
        ]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [2, 4, 5]

    def test_layers_block_type_jamba(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        cfg = type("C", (), {})()
        cfg.layers_block_type = ["mamba", "attention", "mamba", "attention"]
        mc = self._fake_model_config(text_cfg=cfg)
        assert _get_full_attention_layer_indices(mc) == [1, 3]

    def test_attn_type_list_minimax(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        hf = type("C", (), {})()
        hf.attn_type_list = [0, 1, 0, 1, 1]
        mc = self._fake_model_config(hf_cfg=hf)
        assert _get_full_attention_layer_indices(mc) == [1, 3, 4]

    def test_no_hybrid_hints_returns_empty(self):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            _get_full_attention_layer_indices,
        )

        mc = self._fake_model_config()
        assert _get_full_attention_layer_indices(mc) == []


# ============================================================================
# Centroids tests (CPU-only)
# ============================================================================


class TestCentroids:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_centroids_shape(self, bits, expected_n):
        c = get_centroids(128, bits)
        assert c.shape == (expected_n,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        _assert_strictly_sorted(get_centroids(128, bits), "centroids")

    def test_centroids_cached(self):
        c1 = get_centroids(128, 3)
        c2 = get_centroids(128, 3)
        assert c1 is c2, "get_centroids should return cached object"

    def test_centroids_different_dims_not_identical(self):
        c64 = get_centroids(64, 3)
        c128 = get_centroids(128, 3)
        assert not torch.equal(c64, c128)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_symmetric_around_zero(self, bits):
        """N(0, 1/d) is symmetric, so centroids should be ~symmetric."""
        c = get_centroids(128, bits)
        assert abs(c.mean().item()) < 0.01, "Centroids not centered near 0"
        assert abs(c[0].item() + c[-1].item()) < 0.01

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_within_4sigma(self, bits):
        """All centroids should be within ~4 sigma of N(0, 1/d)."""
        sigma = math.sqrt(1.0 / 128)
        c = get_centroids(128, bits)
        for i, val in enumerate(c):
            assert abs(val.item()) < 4 * sigma, (
                f"Centroid {i}={val:.6f} outside 4*sigma={4 * sigma:.6f}"
            )


class TestLloydMax:
    @pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
    def test_solve_shapes(self, bits, expected_n):
        centroids, boundaries = solve_lloyd_max(128, bits)
        assert centroids.shape == (expected_n,)
        assert boundaries.shape == (expected_n - 1,)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        centroids, _ = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(centroids, "centroids")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_sorted(self, bits):
        _, boundaries = solve_lloyd_max(128, bits)
        _assert_strictly_sorted(boundaries, "boundaries")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_between_centroids(self, bits):
        """Each boundary must lie between its adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1], (
                f"Boundary {i}={boundaries[i]:.6f} not between "
                f"c[{i}]={centroids[i]:.6f} and c[{i + 1}]={centroids[i + 1]:.6f}"
            )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_are_midpoints(self, bits):
        """Lloyd-Max boundaries are midpoints of adjacent centroids."""
        centroids, boundaries = solve_lloyd_max(128, bits)
        for i in range(len(boundaries)):
            expected = (centroids[i] + centroids[i + 1]) / 2.0
            assert abs(boundaries[i].item() - expected.item()) < 1e-6

    def test_solve_deterministic(self):
        c1, b1 = solve_lloyd_max(128, 3)
        c2, b2 = solve_lloyd_max(128, 3)
        assert torch.equal(c1, c2)
        assert torch.equal(b1, b2)

    def test_solve_dtype_float32(self):
        centroids, boundaries = solve_lloyd_max(128, 3)
        assert centroids.dtype == torch.float32
        assert boundaries.dtype == torch.float32

    @pytest.mark.parametrize("bits", [3, 4])
    def test_centroids_match_scipy_reference(self, bits):
        """Verify _trapz(n=200) centroids match scipy.integrate.quad reference.

        This ensures our scipy-free trapezoid integration doesn't silently
        drift from the published Lloyd-Max quality.
        """
        pytest.importorskip("scipy")
        from scipy.integrate import quad

        d = 128
        sigma2 = 1.0 / d
        sigma = math.sqrt(sigma2)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
                -x * x / (2 * sigma2)
            )

        n_levels = 2**bits
        lo, hi = -3.5 * sigma, 3.5 * sigma
        ref_centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [
                (ref_centroids[i] + ref_centroids[i + 1]) / 2.0
                for i in range(n_levels - 1)
            ]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = quad(lambda x: x * pdf(x), a, b)
                den, _ = quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else ref_centroids[i])
            if (
                max(abs(new_centroids[i] - ref_centroids[i]) for i in range(n_levels))
                < 1e-10
            ):
                break
            ref_centroids = new_centroids

        # Compare our _trapz centroids against scipy reference
        our_centroids, _ = solve_lloyd_max(d, bits)
        ref_t = torch.tensor(ref_centroids, dtype=torch.float32)
        max_err = (our_centroids - ref_t).abs().max().item()
        # _trapz(n=200) has ~O(h^2) error vs adaptive quad; 1e-3 is tight
        # enough to catch regression while allowing trapezoid approximation.
        assert max_err < 1e-3, (
            f"d={d}, bits={bits}: max centroid error vs scipy = {max_err:.2e}"
        )


# ============================================================================
# Rotation matrix tests (GPU required)
# ============================================================================

GPGPU_AVAILABLE = torch.cuda.is_available() or torch.xpu.is_available()
DEVICE_TYPE = current_platform.device_type


def generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR (test/benchmark only)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestRotationMatrix:
    """Tests for the QR-based rotation (standalone benchmarks only)."""

    @pytest.mark.parametrize("dim", [64, 96, 128, 256])
    def test_rotation_matrix_shape_and_orthogonal(self, dim):
        Pi = generate_rotation_matrix(dim, seed=42, device=DEVICE_TYPE)
        assert Pi.shape == (dim, dim)
        eye = Pi @ Pi.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Pi not orthogonal for dim={dim}"
        )

    def test_rotation_matrix_deterministic(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_rotation_matrix_different_seeds(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=99)
        assert not torch.equal(Pi1, Pi2)

    def test_rotation_matrix_det_is_pm1(self):
        """Orthogonal matrix determinant must be +1 or -1."""
        Pi = generate_rotation_matrix(128, seed=42, device=DEVICE_TYPE)
        det = torch.linalg.det(Pi)
        assert abs(abs(det.item()) - 1.0) < 1e-4


# ============================================================================
# Hadamard rotation tests (serving path: _build_hadamard)
# ============================================================================


def _build_hadamard(d: int, device: str = "cpu") -> torch.Tensor:
    """Reproduce the serving-path Hadamard construction."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device))


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestHadamardRotation:
    """Tests for the Hadamard rotation used in serving."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_orthonormal(self, dim):
        """H must be orthonormal: H @ H^T = I."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        eye = H @ H.T
        assert torch.allclose(eye, torch.eye(dim, device=DEVICE_TYPE), atol=1e-5), (
            f"Hadamard not orthonormal for dim={dim}"
        )

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_hadamard_symmetric(self, dim):
        """Sylvester Hadamard must be symmetric: H = H^T."""
        H = _build_hadamard(dim, DEVICE_TYPE)
        assert torch.allclose(H, H.T, atol=1e-6), (
            f"Hadamard not symmetric for dim={dim}"
        )


# ============================================================================
# Store → Decode round-trip test (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestStoreDecodeRoundTrip:
    """End-to-end: store KV into TQ cache, decode, compare vs fp16 ref."""

    @pytest.mark.parametrize(
        "preset",
        ["turboquant_k8v4", "turboquant_4bit_nc"],
    )
    def test_single_token_roundtrip(self, preset):
        """Store 1 token, decode with query=key, check attention output.

        For a single token with query=key, attention output should equal
        the value (softmax over single key = 1.0). Quantization error
        means we check cosine similarity rather than exact equality.
        """
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            solve_lloyd_max,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        D = 128
        Hk = 4  # num_kv_heads
        Hq = 4  # num_q_heads (no GQA for simplicity)
        B = 1  # single token
        block_size = 16
        num_blocks = 1

        device = torch.device(DEVICE_TYPE)

        # Pure Hadamard rotation (symmetric: H = H^T, so Pi = PiT = H)
        H = _build_hadamard(D, DEVICE_TYPE)
        PiT = H
        Pi = H

        # Generate centroids
        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        # Random K, V
        torch.manual_seed(123)
        key = torch.randn(B, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(B, Hk, D, device=device, dtype=torch.float16)

        # Allocate KV cache
        padded_slot = cfg.slot_size_aligned
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            padded_slot,
            device=device,
            dtype=torch.uint8,
        )
        slot_mapping = torch.tensor([0], device=device, dtype=torch.int32)

        # Store
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
            centroids=c_sorted,
            norm_correction=cfg.norm_correction,
        )

        # Decode: use key as query so attention = softmax([1]) * V = V
        query = key.expand(B, Hq, D).contiguous().to(torch.float16)
        block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([1], device=device, dtype=torch.int32)

        output = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_num_kv_splits=4,
        )

        # With single KV, output should approximate the stored value.
        # Check per-head cosine similarity > threshold.
        out_fp32 = output.float()
        val_fp32 = value.expand(B, Hq, D).float()
        for h in range(Hq):
            cos_sim = torch.nn.functional.cosine_similarity(
                out_fp32[0, h].unsqueeze(0),
                val_fp32[0, h].unsqueeze(0),
            ).item()
            # FP8 keys should be very accurate; MSE keys have more error
            threshold = 0.95 if cfg.key_fp8 else 0.85
            assert cos_sim > threshold, (
                f"Preset {preset} head {h}: cosine_sim={cos_sim:.4f} < {threshold}"
            )


# ============================================================================
# v1 vs v2 decode kernel equivalence (GPU + Triton required)
# ============================================================================


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestDecodeV2Equivalence:
    """Verify the opt-in v2 decode kernel produces the same result as v1.

    v2 is a re-tiled implementation (grouped Q heads, pair LUT, exp2,
    load-halving) reading the same cache layout. For identical inputs
    its output should match v1 up to floating-point rounding noise
    (cosine similarity ~= 1, max abs delta dominated by fp16 tile-order
    differences, never catastrophic disagreement).
    """

    @staticmethod
    def _build_and_store(
        preset: str, Hk: int, D: int, seq_len: int, block_size: int, seed: int
    ):
        """Build inputs and populate KV cache via the production store path."""
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)

        # Pure Hadamard rotation (symmetric, so Pi = PiT).
        H = _build_hadamard(D, DEVICE_TYPE)
        Pi = PiT = H

        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(seed)
        key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        value = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

        num_blocks = (seq_len + block_size - 1) // block_size + 1
        padded_slot = cfg.slot_size_aligned
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            Hk,
            padded_slot,
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
            centroids=c_sorted,
            norm_correction=cfg.norm_correction,
        )
        return cfg, Pi, PiT, centroids, kv_cache, num_blocks

    @staticmethod
    def _run_both_kernels(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, B, Hq, D, seq_len, qseed
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode_v2 import (
            triton_turboquant_decode_attention_v2,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(
                num_blocks,
                device=device,
                dtype=torch.int32,
            )
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

        scale = 1.0 / math.sqrt(D)
        common = dict(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
        )
        out_v1 = triton_turboquant_decode_attention(
            **common,
            max_num_kv_splits=8,
        )
        out_v2 = triton_turboquant_decode_attention_v2(
            **common,
            value_packed_size=cfg.value_packed_size,
            max_seq_len=int(seq_lens.max().item()),
        )
        return query, out_v1, out_v2

    @staticmethod
    def _assert_v1_v2_close(tag, out_v1, out_v2, cos_thr=0.999, abs_thr=0.05):
        a = out_v1.float().flatten()
        b = out_v2.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
        max_abs = (a - b).abs().max().item()
        assert cos_sim > cos_thr, (
            f"{tag} v1/v2 cosine_sim={cos_sim:.6f} below {cos_thr}"
        )
        assert max_abs < abs_thr, f"{tag} v1/v2 max_abs={max_abs:.4e} above {abs_thr}"

    # ------------------------------------------------------------------
    # Tier 1a: kernel-to-kernel equivalence across a wide shape matrix
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (1, 4, 4, 128, 16),  # minimal, GQA=1, short seq
            (4, 8, 2, 128, 64),  # GQA=4
            (4, 16, 2, 128, 1024),  # wide GQA=8, multi-split
            (2, 64, 8, 64, 2048),  # gpt-oss-ish: D=64, GQA=8, 2k ctx
            (2, 32, 4, 128, 4096),  # llama-ish:  D=128, GQA=8, 4k ctx
            (1, 64, 8, 64, 8192),  # long context
        ],
    )
    def test_v1_v2_equivalence(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=16,
            seed=4242,
        )
        _, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        self._assert_v1_v2_close(
            f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]",
            out_v1,
            out_v2,
        )

    # ------------------------------------------------------------------
    # Tier 1b: seq_len edge cases (non-power-of-2, boundaries)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", [1, 2, 17, 33, 127, 129, 513, 1025])
    def test_v1_v2_equivalence_seqlen_edges(self, seq_len):
        # 3-bit preset is the most quantization-lossy -> strongest stress.
        preset = "turboquant_3bit_nc"
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=2,
            D=128,
            seq_len=seq_len,
            block_size=16,
            seed=111,
        )
        _, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=2,
            Hq=8,
            D=128,
            seq_len=seq_len,
            qseed=222,
        )
        self._assert_v1_v2_close(f"[seq_len={seq_len}]", out_v1, out_v2)

    # ------------------------------------------------------------------
    # Tier 1c: FP32 ground-truth absolute accuracy gate
    # ------------------------------------------------------------------
    # Compute true softmax attention on the RAW un-quantized K/V in fp32.
    # Both kernels read the same quantized cache -> same intrinsic quant
    # error vs. the oracle. Assert v2's error is no worse than v1's.

    @staticmethod
    def _reference_attention_fp32(query, raw_k, raw_v, scale):
        B, Hq, D = query.shape
        S, Hk, _ = raw_k.shape
        group = Hq // Hk
        q = query.float()
        k = raw_k.float().repeat_interleave(group, dim=1)
        v = raw_v.float().repeat_interleave(group, dim=1)
        scores = scale * torch.einsum("bhd,shd->bhs", q, k)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("bhs,shd->bhd", probs, v)

    @staticmethod
    def _build_and_store_return_raw(preset, Hk, D, seq_len, block_size, seed):
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )

        device = torch.device(DEVICE_TYPE)
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
        H = _build_hadamard(D, DEVICE_TYPE)
        Pi = PiT = H
        centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
        centroids = centroids.float().to(device)
        c_sorted, _ = centroids.sort()
        midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

        torch.manual_seed(seed)
        raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

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
            raw_k,
            raw_v,
            kv_cache,
            slot_mapping,
            PiT,
            midpoints,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            centroids=c_sorted,
            norm_correction=cfg.norm_correction,
        )
        return (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 8, 2, 128, 128),  # moderate context
            (2, 64, 8, 64, 1024),  # D=64 GQA=8 long-ish
            (1, 32, 4, 128, 2048),  # D=128 GQA=8 longer
        ],
    )
    def test_v2_no_worse_than_v1_vs_fp32_reference(self, preset, B, Hq, Hk, D, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset,
                Hk=Hk,
                D=D,
                seq_len=seq_len,
                block_size=16,
                seed=4242,
            )
        )
        query, out_v1, out_v2 = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        ref = self._reference_attention_fp32(
            query,
            raw_k,
            raw_v,
            scale=1.0 / math.sqrt(D),
        )
        e1 = (out_v1.float() - ref).abs()
        e2 = (out_v2.float() - ref).abs()
        v1_max, v1_mean = e1.max().item(), e1.mean().item()
        v2_max, v2_mean = e2.max().item(), e2.mean().item()

        tag = f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        # Absolute ceiling sanity: attention outputs on N(0,1) inputs are
        # O(1); per-element max-abs error of 1.5 would already mean total
        # loss of signal. Generous bound to catch catastrophic breakage.
        assert v1_max < 1.5, f"{tag} v1 max_err {v1_max:.3f} suspiciously large"
        assert v2_max < 1.5, f"{tag} v2 max_err {v2_max:.3f} suspiciously large"
        # Primary gate: v2 must not be materially worse than v1.
        assert v2_max <= v1_max * 1.10 + 0.02, (
            f"{tag} v2 max_err={v2_max:.4f} exceeds "
            f"v1 max_err={v1_max:.4f} * 1.10 + 0.02"
        )
        assert v2_mean <= v1_mean * 1.10 + 0.005, (
            f"{tag} v2 mean_err={v2_mean:.4f} exceeds "
            f"v1 mean_err={v1_mean:.4f} * 1.10 + 0.005"
        )

    # ------------------------------------------------------------------
    # Tier 1d: v2 determinism (same inputs -> bitwise identical outputs)
    # ------------------------------------------------------------------
    # Guards against any uninitialized memory / race / nondeterministic
    # reduction in the optimized kernel.

    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 16, 2, 128, 1024),  # multi-split decode
            (1, 64, 8, 64, 2048),  # D=64 GQA=8
        ],
    )
    def test_v2_deterministic(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=16,
            seed=4242,
        )
        # Two runs with IDENTICAL inputs. We rebuild query with the same
        # seed to guarantee identical input tensors.
        _, _, out_v2_a = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        _, _, out_v2_b = self._run_both_kernels(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        tag = f"[{preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert torch.equal(out_v2_a, out_v2_b), (
            f"{tag} v2 produced non-bitwise-identical outputs across two "
            f"runs with identical inputs; max diff="
            f"{(out_v2_a.float() - out_v2_b.float()).abs().max().item():.4e}"
        )


# ==========================================================================
# v3: unified prefill+decode kernel (AITER unified_attention backbone with
# TurboQuant K/V dequant inlined). Opt-in path behind VLLM_TQ_DECODE_V3.
# ==========================================================================


class TestDecodeV3Equivalence:
    """v3 (unified) decode path accuracy vs. v1 and vs. an FP32 oracle.

    Shares fixture builders with ``TestDecodeV2Equivalence``: both kernels
    read the same quantized cache, so the intrinsic quant error vs. the
    oracle is identical; we require v3 to not be materially worse than v1.
    """

    # Reuse v2's fixture helpers verbatim (same inputs, same cache).
    _build_and_store = staticmethod(TestDecodeV2Equivalence._build_and_store)
    _build_and_store_return_raw = staticmethod(
        TestDecodeV2Equivalence._build_and_store_return_raw
    )
    _reference_attention_fp32 = staticmethod(
        TestDecodeV2Equivalence._reference_attention_fp32
    )

    @staticmethod
    def _run_v1_v3(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, B, Hq, D, seq_len, qseed
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_decode_attention_v3,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

        scale = 1.0 / math.sqrt(D)
        common = dict(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
        )
        out_v1 = triton_turboquant_decode_attention(**common, max_num_kv_splits=8)
        out_v3 = triton_turboquant_decode_attention_v3(
            **common, value_packed_size=cfg.value_packed_size
        )
        return query, out_v1, out_v3

    @staticmethod
    def _assert_close(tag, a, b, cos_thr=0.999, abs_thr=0.05):
        af = a.float().flatten()
        bf = b.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            af.unsqueeze(0), bf.unsqueeze(0)
        ).item()
        max_abs = (af - bf).abs().max().item()
        assert cos_sim > cos_thr, f"{tag} cosine_sim={cos_sim:.6f} below {cos_thr}"
        assert max_abs < abs_thr, f"{tag} max_abs={max_abs:.4e} above {abs_thr}"

    # ------------------------------------------------------------------
    # Tier 1: v1 <-> v3 equivalence on a compact decode shape matrix
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (1, 4, 4, 128, 16),  # minimal
            (4, 8, 2, 128, 64),  # GQA=4
            (2, 64, 8, 64, 512),  # D=64 GQA=8 (gpt-oss-ish)
            (2, 32, 4, 128, 1024),  # D=128 GQA=8 (llama-ish)
        ],
    )
    def test_v1_v3_equivalence(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
        )
        _, out_v1, out_v3 = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        self._assert_close(
            f"[v3 {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]",
            out_v1,
            out_v3,
        )

    # ------------------------------------------------------------------
    # Tier 2: v3 vs FP32 oracle — must be no materially worse than v1.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 8, 2, 128, 128),  # moderate
            (2, 64, 8, 64, 512),  # D=64 GQA=8
            (1, 32, 4, 128, 1024),  # D=128 GQA=8
        ],
    )
    def test_v3_no_worse_than_v1_vs_fp32_reference(self, preset, B, Hq, Hk, D, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
            )
        )
        query, out_v1, out_v3 = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        ref = self._reference_attention_fp32(
            query, raw_k, raw_v, scale=1.0 / math.sqrt(D)
        )
        e1 = (out_v1.float() - ref).abs()
        e3 = (out_v3.float() - ref).abs()
        v1_max, v1_mean = e1.max().item(), e1.mean().item()
        v3_max, v3_mean = e3.max().item(), e3.mean().item()
        tag = f"[v3 {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert v1_max < 1.5, f"{tag} v1 max_err {v1_max:.3f} suspiciously large"
        assert v3_max < 1.5, f"{tag} v3 max_err {v3_max:.3f} suspiciously large"
        assert v3_max <= v1_max * 1.10 + 0.02, (
            f"{tag} v3 max_err={v3_max:.4f} exceeds "
            f"v1 max_err={v1_max:.4f} * 1.10 + 0.02"
        )
        assert v3_mean <= v1_mean * 1.10 + 0.005, (
            f"{tag} v3 mean_err={v3_mean:.4f} exceeds "
            f"v1 mean_err={v1_mean:.4f} * 1.10 + 0.005"
        )

    # ------------------------------------------------------------------
    # Tier 3: v3 determinism — same inputs must give identical outputs.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        [
            (2, 16, 2, 128, 512),
            (1, 64, 8, 64, 1024),
        ],
    )
    def test_v3_deterministic(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
        )
        _, _, out_v3_a = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        _, _, out_v3_b = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
        )
        tag = f"[v3 {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert torch.equal(out_v3_a, out_v3_b), (
            f"{tag} v3 produced non-bitwise-identical outputs; max diff="
            f"{(out_v3_a.float() - out_v3_b.float()).abs().max().item():.4e}"
        )


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestPrefillV3Equivalence:
    """v3 prefill/chunked accuracy — exercises the BLOCK_M=128 heuristic.

    ``TestDecodeV3Equivalence`` only covers the decode adapter (Q=1 per
    seq, BLOCK_M=16 branch). The launcher's production path for
    prefill/chunked picks BLOCK_M=128, BLOCK_Q=16 for GQA=8 shapes — a
    structurally different code path that was never gated by a strict
    accuracy test. This class closes that gap by running the unified
    launcher directly with Q > 1 and comparing against an fp32 oracle
    on the same quantized cache.
    """

    _build_and_store_return_raw = staticmethod(
        TestDecodeV2Equivalence._build_and_store_return_raw
    )

    @staticmethod
    def _fp32_prefill_oracle(query, raw_k, raw_v, seq_len, Q, scale):
        """Reference fp32 attention for a single-batch prefill.

        Query tokens are treated as KV positions ``[C, C+Q)`` where
        ``C = seq_len - Q``. Causal mask: q_i attends to k_j iff
        ``j <= C + i``.
        """
        # query  : [Q, Hq, D]
        # raw_k/v: [seq_len, Hk, D]
        Hq = query.shape[1]
        Hk = raw_k.shape[1]
        C = seq_len - Q
        kv_group = Hq // Hk
        # Expand KV to match query heads.
        k_exp = raw_k.float().repeat_interleave(kv_group, dim=1)  # [S, Hq, D]
        v_exp = raw_v.float().repeat_interleave(kv_group, dim=1)
        q_f = query.float()  # [Q, Hq, D]
        # Per-head: S = q @ k.T * scale -> [Q, Hq, seq_len]
        scores = torch.einsum("qhd,shd->qhs", q_f, k_exp) * scale
        # Causal mask
        q_pos = torch.arange(Q, device=scores.device) + C  # [Q]
        k_pos = torch.arange(seq_len, device=scores.device)  # [S]
        mask = k_pos[None, :] > q_pos[:, None]  # [Q, S]
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("qhs,shd->qhd", probs, v_exp)  # [Q, Hq, D]
        return out

    @staticmethod
    def _run_v3_prefill(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, Hq, D, seq_len, Q, qseed
    ):
        """Run v3 on a single-batch prefill of ``Q`` tokens attending to a
        pre-stored cache of length ``seq_len``."""
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .contiguous()
        )
        query_start_loc = torch.tensor([0, Q], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)

        out = triton_turboquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            Pi=Pi,
            centroids=centroids,
            scale=scale,
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_query_len=Q,
            max_seq_len=seq_len,
        )
        return query, out

    # ------------------------------------------------------------------
    # Tier 1: v3 prefill vs fp32 oracle on the same quantized cache.
    # A passing test proves the BLOCK_M=128 / BLOCK_Q=16 code path yields
    # the expected quant-aware output (not just self-consistent garbage).
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("preset", ["turboquant_k8v4", "turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "Hq,Hk,D,Q,seq_len",
        [
            # gpt-oss-ish (D=64, GQA=8): short prefill, exercises BLOCK_M=128.
            (64, 8, 64, 128, 256),
            (64, 8, 64, 256, 1024),
            # llama-ish (D=128, GQA=8): same heuristic on the larger head dim.
            (32, 4, 128, 128, 512),
            (32, 4, 128, 256, 1024),
            # Chunked-prefill profile (Q < seq_len; the exact benched shape).
            (64, 8, 64, 64, 1024),
        ],
    )
    def test_v3_prefill_matches_fp32_oracle(self, preset, Hq, Hk, D, Q, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
            )
        )
        query, out_v3 = self._run_v3_prefill(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            Q=Q,
            qseed=7777,
        )
        ref = self._fp32_prefill_oracle(
            query, raw_k, raw_v, seq_len=seq_len, Q=Q, scale=1.0 / math.sqrt(D)
        )

        # Quantization error budget. Thresholds are intentionally loose
        # because TQ 4-bit and 3-bit quantization have real noise — we're
        # gating against CORRECTNESS regressions (catastrophic divergence
        # from the oracle), not bit-exact equivalence. Envelope:
        #   * max_err < 1.5  : same upper bound as the decode oracle test
        #   * cos_sim > 0.98 : catches sign / alignment / head-packing bugs
        #   * mean_err < 0.05: catches systematic offsets
        # For short KVs (seq=256) with 4-bit MSE the cosine can land
        # ~0.992; that is expected quant noise, not a bug.
        err = (out_v3.float() - ref).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        tag = f"[v3-prefill {preset} Hq={Hq} Hk={Hk} D={D} Q={Q} seq={seq_len}]"
        cos_sim = torch.nn.functional.cosine_similarity(
            out_v3.float().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        assert max_err < 1.5, (
            f"{tag} v3 prefill max_err {max_err:.3f} "
            f"(cos_sim={cos_sim:.4f} mean_err={mean_err:.4f}) "
            f"suspiciously large vs oracle"
        )
        assert mean_err < 0.05, (
            f"{tag} v3 prefill mean_err {mean_err:.4f} "
            f"(cos_sim={cos_sim:.4f} max_err={max_err:.4f}) "
            f"exceeds 5% -- likely a kernel bug, not quant noise"
        )
        assert cos_sim > 0.98, (
            f"{tag} v3 prefill cosine_sim={cos_sim:.6f} below 0.98 "
            f"(max_err={max_err:.4f} mean_err={mean_err:.4f})"
        )

    # ------------------------------------------------------------------
    # Tier 2: v3 prefill determinism.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,Hq,Hk,D,Q,seq_len",
        [
            ("turboquant_4bit_nc", 64, 8, 64, 128, 512),
            ("turboquant_4bit_nc", 32, 4, 128, 256, 1024),
        ],
    )
    def test_v3_prefill_deterministic(self, preset, Hq, Hk, D, Q, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, _, _) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
            )
        )
        _, out_a = self._run_v3_prefill(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            Q=Q,
            qseed=13131,
        )
        _, out_b = self._run_v3_prefill(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            Q=Q,
            qseed=13131,
        )
        tag = f"[v3-prefill {preset} Hq={Hq} Hk={Hk} D={D} Q={Q} seq={seq_len}]"
        assert torch.equal(out_a, out_b), (
            f"{tag} v3 prefill produced non-bitwise-identical outputs; "
            f"max diff={(out_a.float() - out_b.float()).abs().max().item():.4e}"
        )

    # ------------------------------------------------------------------
    # Tier 3: LARGE prefill shapes (Q=512, 1024) — confirm BLOCK_M=128
    # path stays correct as Q scales beyond the tier-1 range.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,Hq,Hk,D,Q,seq_len",
        [
            # gpt-oss-ish with larger Q
            ("turboquant_4bit_nc", 64, 8, 64, 512, 1024),
            ("turboquant_4bit_nc", 64, 8, 64, 1024, 2048),
            # llama-ish with larger Q
            ("turboquant_4bit_nc", 32, 4, 128, 512, 1024),
            # Also the k8v4 preset at larger Q (FP8 key path)
            ("turboquant_k8v4", 64, 8, 64, 512, 1024),
        ],
    )
    def test_v3_large_prefill_matches_oracle(self, preset, Hq, Hk, D, Q, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
            )
        )
        query, out_v3 = self._run_v3_prefill(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            Q=Q,
            qseed=7777,
        )
        ref = self._fp32_prefill_oracle(
            query,
            raw_k,
            raw_v,
            seq_len=seq_len,
            Q=Q,
            scale=1.0 / math.sqrt(D),
        )
        err = (out_v3.float() - ref).abs()
        max_err, mean_err = err.max().item(), err.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_v3.float().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        tag = f"[v3-large {preset} Hq={Hq} Hk={Hk} D={D} Q={Q} seq={seq_len}]"
        # Longer KVs average out more quant noise per output element, so
        # we expect tighter cosine than short-KV tier-1 (typically > 0.995).
        assert max_err < 1.5, f"{tag} max_err={max_err:.3f}"
        assert mean_err < 0.05, f"{tag} mean_err={mean_err:.4f}"
        assert cos_sim > 0.99, (
            f"{tag} cos_sim={cos_sim:.6f} (max_err={max_err:.4f} "
            f"mean_err={mean_err:.4f})"
        )


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV3TwoDThreeDEquivalence:
    """2D vs 3D split-KV kernel equivalence.

    Production dispatch routes pure-decode with ``max_seq_len >= 1024`` to
    the 3D kernel + ``reduce_segments``. The 2D kernel can be forced via
    ``force_2d=True``. Both paths must produce numerically equivalent
    output on the same input (small fp32 reassociation noise only; the
    math is the same). This test gates the 3D path independently of the
    quantization oracle — if the 3D kernel has a bug it would disagree
    with the 2D baseline well before either disagrees with an fp32 oracle.
    """

    _build_and_store = staticmethod(TestDecodeV2Equivalence._build_and_store)

    @staticmethod
    def _run_v3(
        cfg,
        Pi,
        PiT,
        centroids,
        kv_cache,
        num_blocks,
        B,
        Hq,
        D,
        seq_len,
        qseed,
        force_2d,
    ):
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        # Decode shape: one token per seq.
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        query_start_loc = torch.arange(B + 1, device=device, dtype=torch.int32) * 1
        # Decode adapter expects [B, Hq, D] but launcher expects [N, Hq, D].
        # For decode N = B * 1 = B; reshape passthrough.
        return triton_turboquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
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

    @pytest.mark.parametrize(
        "preset,B,Hq,Hk,D,seq_len",
        [
            # seq_len >= 1024 triggers 3D path by default; force_2d flips it.
            ("turboquant_4bit_nc", 1, 64, 8, 64, 1024),
            ("turboquant_4bit_nc", 1, 64, 8, 64, 2048),
            ("turboquant_4bit_nc", 4, 64, 8, 64, 2048),
            ("turboquant_4bit_nc", 1, 32, 4, 128, 1024),
            ("turboquant_4bit_nc", 1, 32, 4, 128, 4096),
            ("turboquant_k8v4", 1, 64, 8, 64, 2048),
        ],
    )
    def test_2d_3d_equivalent(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset, Hk=Hk, D=D, seq_len=seq_len, block_size=16, seed=4242
        )
        out_2d = self._run_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
            force_2d=True,
        )
        out_3d = self._run_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
            force_2d=False,
        )
        # 3D does an extra per-segment reduction in fp32; expect small
        # reassociation noise but essentially the same output.
        diff = (out_2d.float() - out_3d.float()).abs()
        max_d, mean_d = diff.max().item(), diff.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            out_2d.float().flatten().unsqueeze(0),
            out_3d.float().flatten().unsqueeze(0),
        ).item()
        tag = f"[2D-vs-3D {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        # Bounds reflect pure fp32 reassociation noise from reduce_segments,
        # not quantization noise (both paths share the same cache).
        assert max_d < 5e-3, f"{tag} max_d={max_d:.4e}"
        assert mean_d < 5e-4, f"{tag} mean_d={mean_d:.4e}"
        assert cos > 0.99999, f"{tag} cos={cos:.8f}"


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV3MixedBatch:
    """Mixed-batch correctness: B>1 with per-sequence seq_lens varying.

    Production inference interleaves sequences of different lengths in
    one launch. The kernel looks up each sequence's block range via
    ``query_start_loc`` and ``seq_lens``; a bug in that per-seq indexing
    would only show up when the lengths differ. Earlier tests all used
    uniform seq_lens.
    """

    _build_and_store_return_raw = staticmethod(
        TestDecodeV2Equivalence._build_and_store_return_raw
    )

    @staticmethod
    def _fp32_oracle_per_seq(queries_per_seq, raw_k, raw_v, seq_lens_list, scale):
        """Compute per-sequence attention in fp32 and concatenate outputs.

        ``queries_per_seq`` is a list of ``[Qi, Hq, D]`` fp16 tensors.
        ``raw_k/raw_v`` is ``[max_seq_len, Hk, D]`` (we slice per-seq).
        For each seq we apply a causal mask where query token i attends
        to KV positions 0..(Si - Qi + i).
        """
        Hk = raw_k.shape[1]
        outs = []
        for q, Si in zip(queries_per_seq, seq_lens_list):
            Qi, Hq, D = q.shape
            Ci = Si - Qi
            kv_group = Hq // Hk
            k = raw_k[:Si].float().repeat_interleave(kv_group, dim=1)
            v = raw_v[:Si].float().repeat_interleave(kv_group, dim=1)
            s = torch.einsum("qhd,shd->qhs", q.float(), k) * scale
            q_pos = torch.arange(Qi, device=q.device) + Ci
            k_pos = torch.arange(Si, device=q.device)
            mask = k_pos[None, :] > q_pos[:, None]
            s = s.masked_fill(mask[:, None, :], float("-inf"))
            probs = torch.softmax(s, dim=-1)
            out = torch.einsum("qhs,shd->qhd", probs, v)
            outs.append(out)
        return torch.cat(outs, dim=0)

    @staticmethod
    def _run_v3_mixed(
        cfg,
        Pi,
        PiT,
        centroids,
        kv_cache,
        num_blocks,
        queries_per_seq,
        seq_lens_list,
        D,
        qseed,
    ):
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )

        device = torch.device(DEVICE_TYPE)
        B = len(seq_lens_list)
        qs = torch.cat(queries_per_seq, dim=0).contiguous()
        # All seqs reuse the same physical block_table (same cache), each
        # seeing the first seq_lens_list[b] tokens of it. This isolates
        # per-seq indexing without needing multiple physical caches.
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        Q_per = torch.tensor(
            [q.shape[0] for q in queries_per_seq],
            device=device,
            dtype=torch.int32,
        )
        query_start_loc = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.cumsum(Q_per, dim=0).to(torch.int32),
            ]
        )
        seq_lens = torch.tensor(
            seq_lens_list,
            device=device,
            dtype=torch.int32,
        )
        max_q = int(Q_per.max().item())
        max_s = int(max(seq_lens_list))
        return triton_turboquant_unified_attention(
            query=qs,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_query_len=max_q,
            max_seq_len=max_s,
        )

    # ------------------------------------------------------------------
    # Mixed-batch DECODE (Q=1 per seq, different seq_lens). Exercises
    # per-seq indexing in the BLOCK_M=16 / decode code path.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,Hq,Hk,D,seq_lens_list",
        [
            ("turboquant_4bit_nc", 64, 8, 64, [256, 1024, 512]),
            ("turboquant_4bit_nc", 64, 8, 64, [1024, 2048, 4096, 1024]),
            ("turboquant_4bit_nc", 32, 4, 128, [512, 1024, 2048]),
            ("turboquant_k8v4", 64, 8, 64, [2048, 1024]),
        ],
    )
    def test_mixed_batch_decode(self, preset, Hq, Hk, D, seq_lens_list):
        max_seq = max(seq_lens_list)
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=max_seq, block_size=16, seed=4242
            )
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        queries_per_seq = [
            torch.randn(1, Hq, D, device=device, dtype=torch.float16)
            for _ in seq_lens_list
        ]
        out_v3 = self._run_v3_mixed(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            queries_per_seq,
            seq_lens_list,
            D=D,
            qseed=7777,
        )
        ref = self._fp32_oracle_per_seq(
            queries_per_seq,
            raw_k,
            raw_v,
            seq_lens_list,
            scale=1.0 / math.sqrt(D),
        )
        err = (out_v3.float() - ref).abs()
        max_err, mean_err = err.max().item(), err.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            out_v3.float().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        tag = f"[mixed-decode {preset} Hq={Hq} Hk={Hk} D={D} seqs={seq_lens_list}]"
        assert max_err < 1.5, f"{tag} max_err={max_err:.3f}"
        assert mean_err < 0.05, f"{tag} mean_err={mean_err:.4f}"
        assert cos > 0.98, (
            f"{tag} cos={cos:.4f} (max_err={max_err:.4f} mean_err={mean_err:.4f})"
        )

    # ------------------------------------------------------------------
    # Mixed-batch PREFILL (Q>1 per seq, different Q/seq_lens). Exercises
    # per-seq indexing in the BLOCK_M=128 / prefill code path.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,Hq,Hk,D,Q_seq_pairs",
        [
            # Mixed chunked prefill: different Q and different context per seq.
            ("turboquant_4bit_nc", 64, 8, 64, [(64, 256), (128, 512), (256, 1024)]),
            ("turboquant_4bit_nc", 32, 4, 128, [(128, 512), (64, 1024)]),
            ("turboquant_k8v4", 64, 8, 64, [(128, 512), (256, 1024)]),
        ],
    )
    def test_mixed_batch_prefill(self, preset, Hq, Hk, D, Q_seq_pairs):
        seq_lens_list = [s for _, s in Q_seq_pairs]
        q_list = [q for q, _ in Q_seq_pairs]
        max_seq = max(seq_lens_list)
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset, Hk=Hk, D=D, seq_len=max_seq, block_size=16, seed=4242
            )
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        queries_per_seq = [
            torch.randn(q, Hq, D, device=device, dtype=torch.float16) for q in q_list
        ]
        out_v3 = self._run_v3_mixed(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            queries_per_seq,
            seq_lens_list,
            D=D,
            qseed=7777,
        )
        ref = self._fp32_oracle_per_seq(
            queries_per_seq,
            raw_k,
            raw_v,
            seq_lens_list,
            scale=1.0 / math.sqrt(D),
        )
        err = (out_v3.float() - ref).abs()
        max_err, mean_err = err.max().item(), err.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            out_v3.float().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0),
        ).item()
        tag = f"[mixed-prefill {preset} Hq={Hq} Hk={Hk} D={D} Q_seq={Q_seq_pairs}]"
        assert max_err < 1.5, f"{tag} max_err={max_err:.3f}"
        assert mean_err < 0.05, f"{tag} mean_err={mean_err:.4f}"
        assert cos > 0.98, (
            f"{tag} cos={cos:.4f} (max_err={max_err:.4f} mean_err={mean_err:.4f})"
        )


# ============================================================================
# Approach 1 — v1 ↔ v3 TIGHT equivalence (Opt#1+#2+#3 regression guard)
# ============================================================================
#
# Thresholds here are an order of magnitude tighter than
# ``TestDecodeV3Equivalence`` (which uses the TQ-vs-reference quant budget
# of cos>0.999 / abs<0.05). The tight values below are locked from the
# ``measure_v1_v3_drift.py`` sweep (120 cells) using
# ``threshold = 1.2 × p99_observed`` with an analytical BF16-ULP floor.
#
# Goal: catch *any* regression introduced by Opt#1 (norm-correction
# baking), Opt#2 (uint16 wide metadata loads), or Opt#3 (SoA relayout),
# without rejecting the legitimate BF16 rounding noise they produce.
# See /workspace/benchmarks/correctness/APPROACH1_V1_V3_EQUIVALENCE.md
# for the design and the measurement CSV.


_TIER_A_THRESHOLDS = {
    # query_dtype -> (cos_min, max_abs, mean_abs, p99_abs, snr_max)
    torch.float16: (0.999990, 1.5e-3, 2.0e-4, 8.0e-4, 0.20),
    torch.bfloat16: (0.999900, 6.0e-3, 1.0e-3, 3.0e-3, 0.15),
}

# Tier D — analytical ULP-bound ratios. max/ulp and std/ulp must both
# stay under 2× the tile-level BF16-ULP envelope. Observed max across
# the 120-cell sweep was 0.86 and 0.156 respectively; 2× gives comfortable
# headroom while still detecting a systematic ~3× blow-up.
_TIER_D_MAX_OVER_ULP = 2.0
_TIER_D_STD_OVER_ULP = 2.0
_BLOCK_N = 64  # v3 decode K-tile width


def _drift_stats(out_v1: torch.Tensor, out_v3: torch.Tensor, seq_len: int) -> dict:
    """Compute every stat used by Tiers A and D."""
    a = out_v1.float()
    b = out_v3.float()
    d = b - a
    d_abs = d.abs()

    cos_sim = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    max_abs = d_abs.max().item()
    mean_abs = d_abs.mean().item()
    flat = d_abs.flatten()
    if flat.numel() > 100_000:
        idx = torch.randperm(flat.numel(), device=flat.device)[:100_000]
        p99_abs = flat[idx].quantile(0.99).item()
    else:
        p99_abs = flat.quantile(0.99).item()

    mean_d = d.mean().item()
    std_d = d.std().item() if d.numel() > 1 else 0.0
    snr = abs(mean_d) / std_d if std_d > 0 else 0.0

    # Analytical BF16-ULP ceiling (Approach-1 doc §2.2).
    n_tiles = max(1, (seq_len + _BLOCK_N - 1) // _BLOCK_N)
    out_scale = a.abs().max().item()
    ulp_bound = max(1e-12, n_tiles * (2.0**-7) * out_scale)

    return dict(
        cos_sim=cos_sim,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        mean_d=mean_d,
        std_d=std_d,
        snr=snr,
        ulp_bound=ulp_bound,
        max_over_ulp=max_abs / ulp_bound,
        std_over_ulp=std_d / ulp_bound,
    )


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV1V3TightEquivalence:
    """Tight v1 ↔ v3 decode equivalence + ULP-bounded drift shape.

    This class asserts v3 behaves like v1 up to BF16 rounding — not up to
    the TQ quant budget. Four tiers:

      A. magnitude    — cos/max/mean/p99 thresholds per dtype
      B. determinism  — v3(x) == v3(x) bit-for-bit
      C. FP32 oracle  — v3 not worse than v1 by >5% vs ground truth
      D. drift shape  — mean/std/snr within analytical BF16-ULP envelope

    No prefill equivalent: v1 has no quantized prefill kernel (first-chunk
    prefill in v1 uses flash_attn on raw K/V, pre-quant), so the
    comparison is not defined for prefill. Prefill correctness stays in
    ``TestPrefillV3Equivalence``.
    """

    _build_and_store = staticmethod(TestDecodeV2Equivalence._build_and_store)
    _build_and_store_return_raw = staticmethod(
        TestDecodeV2Equivalence._build_and_store_return_raw
    )
    _reference_attention_fp32 = staticmethod(
        TestDecodeV2Equivalence._reference_attention_fp32
    )

    @staticmethod
    def _run_v1_v3(
        cfg,
        Pi,
        PiT,
        centroids,
        kv_cache,
        num_blocks,
        B,
        Hq,
        D,
        seq_len,
        qseed,
        query_dtype,
    ):
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_decode_attention_v3,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(B, Hq, D, device=device, dtype=query_dtype)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        common = dict(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
        )
        out_v1 = triton_turboquant_decode_attention(**common, max_num_kv_splits=8)
        out_v3 = triton_turboquant_decode_attention_v3(
            **common, value_packed_size=cfg.value_packed_size
        )
        return out_v1, out_v3

    # ------------------------------------------------------------------
    # TIER A — magnitude close to v1 per dtype.
    # Parametrization covers both presets, FP16/BF16 queries, both
    # block_sizes (16 and 64), GQA 1/4/8, D ∈ {64,128}, seq up to 8k,
    # and batches up to 32.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,B,Hq,Hk,D,seq_len,block_size",
        [
            ("turboquant_4bit_nc", 1, 4, 4, 128, 64, 16),
            ("turboquant_4bit_nc", 1, 4, 4, 128, 64, 64),
            ("turboquant_4bit_nc", 4, 8, 2, 128, 512, 16),
            ("turboquant_4bit_nc", 4, 8, 2, 128, 2048, 64),
            ("turboquant_4bit_nc", 2, 64, 8, 64, 512, 16),
            ("turboquant_4bit_nc", 2, 64, 8, 64, 2048, 64),
            ("turboquant_4bit_nc", 1, 64, 8, 64, 8192, 64),  # long ctx
            ("turboquant_4bit_nc", 2, 32, 4, 128, 1024, 64),
            ("turboquant_4bit_nc", 2, 32, 4, 128, 4096, 64),
            ("turboquant_4bit_nc", 32, 4, 4, 128, 256, 64),  # big batch
            ("turboquant_k8v4", 2, 64, 8, 64, 2048, 64),
            ("turboquant_k8v4", 2, 32, 4, 128, 1024, 64),
        ],
    )
    @pytest.mark.parametrize(
        "query_dtype",
        [torch.float16, torch.bfloat16],
        ids=["qfp16", "qbf16"],
    )
    def test_v1_v3_decode_tight(
        self, preset, B, Hq, Hk, D, seq_len, block_size, query_dtype
    ):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=block_size,
            seed=4242,
        )
        out_v1, out_v3 = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
            query_dtype=query_dtype,
        )
        st = _drift_stats(out_v1, out_v3, seq_len)
        cos_min, max_ceil, mean_ceil, p99_ceil, snr_ceil = _TIER_A_THRESHOLDS[
            query_dtype
        ]
        tag = (
            f"[TIER-A {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"bs={block_size} {str(query_dtype).replace('torch.', '')}]"
        )
        assert st["cos_sim"] > cos_min, (
            f"{tag} cos_sim={st['cos_sim']:.6f} ≤ {cos_min} "
            f"(max_abs={st['max_abs']:.3e} mean_abs={st['mean_abs']:.3e})"
        )
        assert st["max_abs"] < max_ceil, (
            f"{tag} max_abs={st['max_abs']:.3e} ≥ {max_ceil} "
            f"(cos_sim={st['cos_sim']:.6f})"
        )
        assert st["mean_abs"] < mean_ceil, (
            f"{tag} mean_abs={st['mean_abs']:.3e} ≥ {mean_ceil}"
        )
        assert st["p99_abs"] < p99_ceil, (
            f"{tag} p99_abs={st['p99_abs']:.3e} ≥ {p99_ceil}"
        )
        assert st["snr"] < snr_ceil, (
            f"{tag} |mean|/std={st['snr']:.4f} ≥ {snr_ceil} "
            f"(drift looks biased, not noise-shaped)"
        )

    # ------------------------------------------------------------------
    # TIER B — v3 determinism. Back-to-back v3 calls on identical inputs
    # must produce bit-identical outputs.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,B,Hq,Hk,D,seq_len",
        [
            ("turboquant_4bit_nc", 2, 16, 2, 128, 512),
            ("turboquant_4bit_nc", 1, 64, 8, 64, 1024),
            ("turboquant_k8v4", 1, 64, 8, 64, 1024),
        ],
    )
    def test_v3_deterministic(self, preset, B, Hq, Hk, D, seq_len):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=16,
            seed=4242,
        )
        _, out_a = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
            query_dtype=torch.float16,
        )
        _, out_b = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=13131,
            query_dtype=torch.float16,
        )
        tag = f"[TIER-B {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert torch.equal(out_a, out_b), (
            f"{tag} v3 non-deterministic; "
            f"max diff={(out_a.float() - out_b.float()).abs().max().item():.4e}"
        )

    # ------------------------------------------------------------------
    # TIER C — no regression vs FP32 oracle (tightened from 10 % to 5 %).
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,B,Hq,Hk,D,seq_len",
        [
            ("turboquant_4bit_nc", 2, 8, 2, 128, 128),
            ("turboquant_4bit_nc", 2, 64, 8, 64, 512),
            ("turboquant_4bit_nc", 1, 32, 4, 128, 1024),
            ("turboquant_k8v4", 2, 64, 8, 64, 512),
        ],
    )
    def test_v3_no_regression_vs_fp32(self, preset, B, Hq, Hk, D, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                preset,
                Hk=Hk,
                D=D,
                seq_len=seq_len,
                block_size=16,
                seed=4242,
            )
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        out_v1, out_v3 = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
            query_dtype=torch.float16,
        )
        ref = self._reference_attention_fp32(
            query, raw_k, raw_v, scale=1.0 / math.sqrt(D)
        )
        e1 = (out_v1.float() - ref).abs()
        e3 = (out_v3.float() - ref).abs()
        v1_max, v1_mean = e1.max().item(), e1.mean().item()
        v3_max, v3_mean = e3.max().item(), e3.mean().item()
        tag = f"[TIER-C {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert v1_max < 1.5, f"{tag} v1 max_err {v1_max:.3f} suspiciously large"
        assert v3_max < 1.5, f"{tag} v3 max_err {v3_max:.3f} suspiciously large"
        assert v3_max <= v1_max * 1.05 + 0.01, (
            f"{tag} v3 max_err={v3_max:.4f} exceeds "
            f"v1 max_err={v1_max:.4f} * 1.05 + 0.01"
        )
        assert v3_mean <= v1_mean * 1.05 + 0.005, (
            f"{tag} v3 mean_err={v3_mean:.4f} exceeds "
            f"v1 mean_err={v1_mean:.4f} * 1.05 + 0.005"
        )

    # ------------------------------------------------------------------
    # TIER D — shape of drift (not just magnitude). Asserts v3 vs v1
    # delta looks like BF16-ULP-bounded noise, not systematic bias.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "preset,B,Hq,Hk,D,seq_len,block_size",
        [
            ("turboquant_4bit_nc", 1, 4, 4, 128, 64, 16),
            ("turboquant_4bit_nc", 2, 64, 8, 64, 512, 16),
            ("turboquant_4bit_nc", 2, 64, 8, 64, 2048, 64),
            ("turboquant_4bit_nc", 2, 32, 4, 128, 4096, 64),
            ("turboquant_4bit_nc", 1, 64, 8, 64, 8192, 64),
            ("turboquant_k8v4", 2, 64, 8, 64, 2048, 64),
        ],
    )
    @pytest.mark.parametrize(
        "query_dtype",
        [torch.float16, torch.bfloat16],
        ids=["qfp16", "qbf16"],
    )
    def test_v1_v3_drift_matches_ulp_bound(
        self, preset, B, Hq, Hk, D, seq_len, block_size, query_dtype
    ):
        cfg, Pi, PiT, centroids, kv_cache, num_blocks = self._build_and_store(
            preset,
            Hk=Hk,
            D=D,
            seq_len=seq_len,
            block_size=block_size,
            seed=4242,
        )
        out_v1, out_v3 = self._run_v1_v3(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
            query_dtype=query_dtype,
        )
        st = _drift_stats(out_v1, out_v3, seq_len)
        tag = (
            f"[TIER-D {preset} B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"bs={block_size} {str(query_dtype).replace('torch.', '')}]"
        )
        assert st["max_over_ulp"] < _TIER_D_MAX_OVER_ULP, (
            f"{tag} max_abs / ulp_bound = {st['max_over_ulp']:.3f} "
            f">= {_TIER_D_MAX_OVER_ULP} — drift larger than BF16 rounding "
            f"can explain (max_abs={st['max_abs']:.3e}, "
            f"ulp_bound={st['ulp_bound']:.3e})"
        )
        assert st["std_over_ulp"] < _TIER_D_STD_OVER_ULP, (
            f"{tag} std_d / ulp_bound = {st['std_over_ulp']:.3f} "
            f">= {_TIER_D_STD_OVER_ULP} — drift spread exceeds envelope "
            f"(std_d={st['std_d']:.3e}, ulp_bound={st['ulp_bound']:.3e})"
        )
        ulp_bound_mean = 0.5 * st["ulp_bound"]
        assert abs(st["mean_d"]) < ulp_bound_mean, (
            f"{tag} |mean(d)|={abs(st['mean_d']):.3e} >= "
            f"0.5 × ulp_bound={ulp_bound_mean:.3e} — drift is biased"
        )


# ============================================================================
# Approach 2 — v3 vs FP32 reference attention (quantization-budget gate)
# ============================================================================
#
# Verifies that v3 (unified decode + prefill) stays inside the legitimate
# quantization-loss budget for ``turboquant_4bit_nc`` — the sole preset on
# every production benchmark path. This is *absolute* accuracy vs the
# ground-truth FP32 attention computed on the raw pre-quantization K/V,
# complementing Approach 1 (which asserts v3 ≈ v1 in BF16-ULP terms).
#
# Design doc: /workspace/benchmarks/correctness/APPROACH2_V3_VS_REFERENCE.md
# Measurement CSV: /workspace/benchmarks/correctness/v3_vs_reference_measurements.csv
#
# Primary metric: per-(B,H) 5th-percentile cosine similarity (``cos_p5``).
# This was chosen over ``cos_min`` because ``cos_p5`` lands in a tight
# 0.9859–0.9892 band across the full 60-cell sweep, while ``cos_min``
# spreads from 0.936 to 0.985 purely because larger ``N_slots = B·Hq·Q``
# gives more chances for one unlucky slot. ``cos_min`` remains a loose
# defense-in-depth gate to catch catastrophic single-slot failures.

# Decode: Q=1 per seq, N_slots = B × Hq
_BUDGET_4BIT_NC_DECODE = dict(
    cos_p5_min=0.983,  # 5th percentile of per-(B,H) cos similarity
    cos_min_min=0.95,  # worst single (B,H) — catches catastrophic failures
    relerr_max=0.35,  # max ||Δ||/||ref|| per (B,H); 1.40× worst observed
    abs_max=0.12,  # max |out - ref|; 1.43× worst observed (0.084)
)

# Prefill: Q = seq_len/4, N_slots = B × Q × Hq (much larger → looser cos_min)
_BUDGET_4BIT_NC_PREFILL = dict(
    cos_p5_min=0.983,  # same percentile threshold (percentile is robust)
    cos_min_min=0.90,  # looser min because more slots => more outlier chance
    relerr_max=0.45,  # 1.27× worst observed
    abs_max=0.25,  # 1.28× worst observed (0.195)
)


def _per_bh_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Compute per-(token-slot × head) cos and relerr metrics.

    ``out`` and ``ref`` share leading dims. For decode both are ``[B,Hq,D]``
    (so slot = batch). For prefill both are ``[B*Q, Hq, D]`` (so slot =
    token-within-the-flat-packing). In either case each (slot, head) pair
    is one query vector; we flatten to ``[N_slots, D]`` and compute cos /
    rel-err per row, then aggregate.
    """
    a = out.float()
    b = ref.float()
    N, Hq, D = a.shape
    a_flat = a.reshape(N * Hq, D)
    b_flat = b.reshape(N * Hq, D)

    eps = 1e-12
    a_norm = a_flat.norm(dim=-1).clamp_min(eps)
    b_norm = b_flat.norm(dim=-1).clamp_min(eps)
    cos_bh = (a_flat * b_flat).sum(dim=-1) / (a_norm * b_norm)

    diff_norm = (a_flat - b_flat).norm(dim=-1)
    relerr_bh = diff_norm / b_norm

    abs_err = (a - b).abs()
    return dict(
        cos_min=cos_bh.min().item(),
        cos_p5=torch.quantile(cos_bh.float(), 0.05).item(),
        cos_mean=cos_bh.mean().item(),
        relerr_max=relerr_bh.max().item(),
        relerr_mean=relerr_bh.mean().item(),
        abs_max=abs_err.max().item(),
        abs_mean=abs_err.mean().item(),
        n_slots=cos_bh.numel(),
    )


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV3VsReference:
    """v3 absolute accuracy vs FP32 standard attention on ``turboquant_4bit_nc``.

    Three tiers:

    * **α (per-query budget)** — ``min_{b,h} cos ≥ loose`` and
      ``p5_{b,h} cos ≥ 0.983`` (primary gate). Also ``relerr_max`` and
      ``abs_max`` ceilings for defense in depth. Runs on both decode
      (Q=1) and prefill (Q=seq_len/4).
    * **β (shape scaling)** — decode ``abs_max`` must shrink by ≥ 25 %
      per 4× increase in seq_len (expected ~50 %). Catches bugs whose
      error grows or stays flat with context length.
    * **γ (multi-seed robust)** — worst ``cos_p5`` across 5 seeds still
      meets the Tier-α threshold. Catches threshold overfitting to a
      single lucky seed.

    All thresholds are locked from an empirical 60-cell sweep; see the
    design doc for derivation.
    """

    # Reuse infrastructure from the existing helpers.
    _build_and_store_return_raw = staticmethod(
        TestDecodeV2Equivalence._build_and_store_return_raw
    )
    _reference_attention_fp32 = staticmethod(
        TestDecodeV2Equivalence._reference_attention_fp32
    )

    # ------------------------------------------------------------------
    # Kernel runners (decode + prefill) — mirror measurement script.
    # ------------------------------------------------------------------

    @staticmethod
    def _run_v3_decode(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, B, Hq, D, seq_len, qseed
    ):
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_decode_attention_v3,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        out = triton_turboquant_decode_attention_v3(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
        )
        return query, out

    @staticmethod
    def _run_v3_prefill(
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, B, Hq, D, seq_len, Q, qseed
    ):
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(qseed)
        # Packed query: [B*Q, Hq, D]. All B seqs share the same KV cache.
        query = torch.randn(B * Q, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        query_start_loc = torch.arange(B + 1, device=device, dtype=torch.int32) * Q
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        out = triton_turboquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            value_packed_size=cfg.value_packed_size,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=PiT,
            max_query_len=Q,
            max_seq_len=seq_len,
        )
        return query, out

    @staticmethod
    def _reference_prefill_fp32(query, raw_k, raw_v, B, Q, seq_len, scale):
        """FP32 causal prefill oracle for B seqs × Q tokens sharing one KV."""
        Hq = query.shape[1]
        Hk = raw_k.shape[1]
        C = seq_len - Q
        group = Hq // Hk
        k_exp = raw_k.float().repeat_interleave(group, dim=1)
        v_exp = raw_v.float().repeat_interleave(group, dim=1)
        out = []
        for b in range(B):
            q_f = query[b * Q : (b + 1) * Q].float()
            scores = torch.einsum("qhd,shd->qhs", q_f, k_exp) * scale
            q_pos = torch.arange(Q, device=scores.device) + C
            k_pos = torch.arange(seq_len, device=scores.device)
            mask = k_pos[None, :] > q_pos[:, None]
            scores = scores.masked_fill(mask[:, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out.append(torch.einsum("qhs,shd->qhd", probs, v_exp))
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # Shape matrix shared across tiers.
    # ------------------------------------------------------------------
    # (B, Hq, Hk, D, seq_len) — mirrors the measurement sweep.
    _ALPHA_SHAPES = [
        (2, 64, 8, 64, 256),
        (2, 64, 8, 64, 1024),
        (2, 64, 8, 64, 4096),
        (2, 32, 4, 128, 256),
        (2, 32, 4, 128, 1024),
        (2, 32, 4, 128, 4096),
    ]

    # ------------------------------------------------------------------
    # TIER α — per-query budget (decode + prefill).
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        _ALPHA_SHAPES,
        ids=[f"B{B}Hq{Hq}Hk{Hk}D{D}S{S}" for (B, Hq, Hk, D, S) in _ALPHA_SHAPES],
    )
    def test_v3_decode_per_query_budget(self, B, Hq, Hk, D, seq_len):
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                "turboquant_4bit_nc",
                Hk=Hk,
                D=D,
                seq_len=seq_len,
                block_size=16,
                seed=4242,
            )
        )
        query, out = self._run_v3_decode(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            qseed=7777,
        )
        ref = self._reference_attention_fp32(
            query,
            raw_k,
            raw_v,
            scale=1.0 / math.sqrt(D),
        )
        st = _per_bh_metrics(out, ref)
        b = _BUDGET_4BIT_NC_DECODE
        tag = (
            f"[α-decode B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"N_slots={st['n_slots']}]"
        )
        assert st["cos_p5"] >= b["cos_p5_min"], (
            f"{tag} cos_p5={st['cos_p5']:.5f} < {b['cos_p5_min']} "
            f"(cos_min={st['cos_min']:.5f}, cos_mean={st['cos_mean']:.5f})"
        )
        assert st["cos_min"] >= b["cos_min_min"], (
            f"{tag} cos_min={st['cos_min']:.5f} < {b['cos_min_min']} "
            f"— catastrophic per-slot failure"
        )
        assert st["relerr_max"] <= b["relerr_max"], (
            f"{tag} relerr_max={st['relerr_max']:.4f} > {b['relerr_max']}"
        )
        assert st["abs_max"] <= b["abs_max"], (
            f"{tag} abs_max={st['abs_max']:.4f} > {b['abs_max']}"
        )

    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        _ALPHA_SHAPES,
        ids=[f"B{B}Hq{Hq}Hk{Hk}D{D}S{S}" for (B, Hq, Hk, D, S) in _ALPHA_SHAPES],
    )
    def test_v3_prefill_per_query_budget(self, B, Hq, Hk, D, seq_len):
        Q = seq_len // 4
        (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
            self._build_and_store_return_raw(
                "turboquant_4bit_nc",
                Hk=Hk,
                D=D,
                seq_len=seq_len,
                block_size=16,
                seed=4242,
            )
        )
        query, out = self._run_v3_prefill(
            cfg,
            Pi,
            PiT,
            centroids,
            kv_cache,
            num_blocks,
            B=B,
            Hq=Hq,
            D=D,
            seq_len=seq_len,
            Q=Q,
            qseed=7777,
        )
        ref = self._reference_prefill_fp32(
            query,
            raw_k,
            raw_v,
            B=B,
            Q=Q,
            seq_len=seq_len,
            scale=1.0 / math.sqrt(D),
        )
        st = _per_bh_metrics(out, ref)
        b = _BUDGET_4BIT_NC_PREFILL
        tag = (
            f"[α-prefill B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"Q={Q} N_slots={st['n_slots']}]"
        )
        assert st["cos_p5"] >= b["cos_p5_min"], (
            f"{tag} cos_p5={st['cos_p5']:.5f} < {b['cos_p5_min']} "
            f"(cos_min={st['cos_min']:.5f}, cos_mean={st['cos_mean']:.5f})"
        )
        assert st["cos_min"] >= b["cos_min_min"], (
            f"{tag} cos_min={st['cos_min']:.5f} < {b['cos_min_min']} "
            f"— catastrophic per-slot failure"
        )
        assert st["relerr_max"] <= b["relerr_max"], (
            f"{tag} relerr_max={st['relerr_max']:.4f} > {b['relerr_max']}"
        )
        assert st["abs_max"] <= b["abs_max"], (
            f"{tag} abs_max={st['abs_max']:.4f} > {b['abs_max']}"
        )

    # ------------------------------------------------------------------
    # TIER β — shape scaling on decode abs_mean.
    # ------------------------------------------------------------------
    # Quant noise averages out as seq_len grows. The claim is about the
    # **typical** error magnitude, so ``abs_mean`` (mean of all output
    # errors) is the right statistic — ``abs_max`` tracks an extreme
    # value which is noisy seed-to-seed, while ``abs_mean`` directly
    # follows the 1/√S averaging prediction.
    #
    # 5 seeds × 3 seq_lens, aggregated by **arithmetic mean** over seeds
    # (smoother than median for small sample sizes).
    #
    # Only decode is gated; prefill has Q=seq_len/4 so N_slots scales
    # with S, which confounds the shape-scaling signal.

    @pytest.mark.parametrize(
        "B,Hq,Hk,D",
        [(2, 64, 8, 64), (2, 32, 4, 128)],
        ids=["D64", "D128"],
    )
    def test_v3_decode_shape_scaling_monotonic(self, B, Hq, Hk, D):
        abs_by_seq: dict[int, list[float]] = {256: [], 1024: [], 4096: []}
        for seed in (1, 2, 3, 4, 5):
            for S in (256, 1024, 4096):
                (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
                    self._build_and_store_return_raw(
                        "turboquant_4bit_nc",
                        Hk=Hk,
                        D=D,
                        seq_len=S,
                        block_size=16,
                        seed=seed,
                    )
                )
                query, out = self._run_v3_decode(
                    cfg,
                    Pi,
                    PiT,
                    centroids,
                    kv_cache,
                    num_blocks,
                    B=B,
                    Hq=Hq,
                    D=D,
                    seq_len=S,
                    qseed=seed + 7777,
                )
                ref = self._reference_attention_fp32(
                    query,
                    raw_k,
                    raw_v,
                    scale=1.0 / math.sqrt(D),
                )
                st = _per_bh_metrics(out, ref)
                abs_by_seq[S].append(st["abs_mean"])

        agg = {S: sum(vs) / len(vs) for S, vs in abs_by_seq.items()}
        tag = f"[β-decode Hq={Hq} Hk={Hk} D={D}]"
        # Require ≥ 25 % decrease per 4× seq_len step. Theoretical 1/√S
        # scaling predicts ~50 % decrease; 25 % is the floor with safety.
        assert agg[1024] <= 0.75 * agg[256], (
            f"{tag} abs_mean(S=1024)={agg[1024]:.5f} > 0.75 × "
            f"abs_mean(S=256)={agg[256]:.5f}; quant noise not averaging out "
            f"(per-seed abs_mean: {abs_by_seq})"
        )
        assert agg[4096] <= 0.75 * agg[1024], (
            f"{tag} abs_mean(S=4096)={agg[4096]:.5f} > 0.75 × "
            f"abs_mean(S=1024)={agg[1024]:.5f}; quant noise not averaging out "
            f"(per-seed abs_mean: {abs_by_seq})"
        )

    # ------------------------------------------------------------------
    # TIER γ — multi-seed robustness.
    # ------------------------------------------------------------------
    # Worst cos_p5 across 5 seeds must still pass the Tier α gate. Kept
    # to a smaller shape set than α to bound suite runtime (5× factor).

    _GAMMA_SHAPES = [
        (2, 64, 8, 64, 1024),  # gpt-oss-ish mid-ctx
        (2, 32, 4, 128, 1024),  # llama/qwen-ish mid-ctx
    ]

    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        _GAMMA_SHAPES,
        ids=[f"B{B}Hq{Hq}Hk{Hk}D{D}S{S}" for (B, Hq, Hk, D, S) in _GAMMA_SHAPES],
    )
    def test_v3_decode_multi_seed_robust(self, B, Hq, Hk, D, seq_len):
        cos_p5_per_seed = []
        for seed in (1, 2, 3, 4, 5):
            (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
                self._build_and_store_return_raw(
                    "turboquant_4bit_nc",
                    Hk=Hk,
                    D=D,
                    seq_len=seq_len,
                    block_size=16,
                    seed=seed,
                )
            )
            query, out = self._run_v3_decode(
                cfg,
                Pi,
                PiT,
                centroids,
                kv_cache,
                num_blocks,
                B=B,
                Hq=Hq,
                D=D,
                seq_len=seq_len,
                qseed=seed + 7777,
            )
            ref = self._reference_attention_fp32(
                query,
                raw_k,
                raw_v,
                scale=1.0 / math.sqrt(D),
            )
            st = _per_bh_metrics(out, ref)
            cos_p5_per_seed.append(st["cos_p5"])
        worst = min(cos_p5_per_seed)
        thr = _BUDGET_4BIT_NC_DECODE["cos_p5_min"]
        tag = f"[γ-decode B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len}]"
        assert worst >= thr, (
            f"{tag} worst cos_p5 across 5 seeds = {worst:.5f} < {thr} "
            f"(all seeds: {[round(v, 5) for v in cos_p5_per_seed]})"
        )

    @pytest.mark.parametrize(
        "B,Hq,Hk,D,seq_len",
        _GAMMA_SHAPES,
        ids=[f"B{B}Hq{Hq}Hk{Hk}D{D}S{S}" for (B, Hq, Hk, D, S) in _GAMMA_SHAPES],
    )
    def test_v3_prefill_multi_seed_robust(self, B, Hq, Hk, D, seq_len):
        Q = seq_len // 4
        cos_p5_per_seed = []
        for seed in (1, 2, 3, 4, 5):
            (cfg, Pi, PiT, centroids, kv_cache, num_blocks, raw_k, raw_v) = (
                self._build_and_store_return_raw(
                    "turboquant_4bit_nc",
                    Hk=Hk,
                    D=D,
                    seq_len=seq_len,
                    block_size=16,
                    seed=seed,
                )
            )
            query, out = self._run_v3_prefill(
                cfg,
                Pi,
                PiT,
                centroids,
                kv_cache,
                num_blocks,
                B=B,
                Hq=Hq,
                D=D,
                seq_len=seq_len,
                Q=Q,
                qseed=seed + 7777,
            )
            ref = self._reference_prefill_fp32(
                query,
                raw_k,
                raw_v,
                B=B,
                Q=Q,
                seq_len=seq_len,
                scale=1.0 / math.sqrt(D),
            )
            st = _per_bh_metrics(out, ref)
            cos_p5_per_seed.append(st["cos_p5"])
        worst = min(cos_p5_per_seed)
        thr = _BUDGET_4BIT_NC_PREFILL["cos_p5_min"]
        tag = f"[γ-prefill B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} Q={Q}]"
        assert worst >= thr, (
            f"{tag} worst cos_p5 across 5 seeds = {worst:.5f} < {thr} "
            f"(all seeds: {[round(v, 5) for v in cos_p5_per_seed]})"
        )
