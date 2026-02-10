"""
Validate `srs_estimator_torch.srs_channel_estimator` against case0 vectors
used by the NumPy reference script (`validate_case0.py`).
"""

import struct
from pathlib import Path

import numpy as np
import torch

from srs_estimator_torch import EstimatorConfig, HopConfig, srs_channel_estimator


def load_expected_entries(path: str):
    """
    Parse resource_grid_reader_spy::expected_entry_t dumped by file_vector.
    Layout inferred from binary:
      uint16 packed_sym_port  (symbol = packed >> 8, port = packed & 0xFF)
      uint16 subcarrier_index (absolute)
      float32 real
      float32 imag
    => 12 bytes per entry.
    Returns list of tuples (symbol, port, sc, value).
    """
    raw = Path(path).read_bytes()
    if len(raw) % 12 != 0:
        raise ValueError(f"{path} size {len(raw)} not multiple of 12 bytes.")
    out = []
    for i in range(0, len(raw), 12):
        packed, sc, re, im = struct.unpack_from("<HHff", raw, i)
        sym = packed >> 8
        port = packed & 0xFF
        out.append((sym, port, sc, re + 1j * im))
    return out


def entries_to_grid(entries, n_sc_total=52 * 12, n_sym_total=14):
    n_layers = max(p for _, p, _, _ in entries) + 1
    grid = np.zeros((n_sc_total, n_sym_total, n_layers), dtype=np.complex128)
    for sym, port, sc, val in entries:
        grid[sc, sym, port] = val
    return grid

# Case 0 로딩
rg_entries = load_expected_entries("./testvector_outputs/port_channel_estimator_test_input_rg0.dat")
ch_entries = load_expected_entries("./testvector_outputs/port_channel_estimator_test_output_ch_est0.dat")
pilots = np.fromfile("./testvector_outputs/port_channel_estimator_test_pilots0.dat", dtype=np.complex64)

# -----------------------------
# Config from port_channel_estimator_test_data.h (case 0)
# -----------------------------
dmrs_symbols = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
dmrs_re_mask = np.array(
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool
).reshape(12, 1)
mask_prbs = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=bool,
)


def main():
    device = "cpu"

    hop1_config = HopConfig(
        DMRSsymbols=torch.as_tensor(dmrs_symbols, device=device),
        DMRSREmask=torch.as_tensor(dmrs_re_mask, device=device),
        PRBstart=40,
        nPRBs=int(mask_prbs.sum()),
        maskPRBs=torch.as_tensor(mask_prbs, device=device),
        startSymbol=0,
        nAllocatedSymbols=14,
    )

    # Hop 2 없음 (single hop)
    hop2_config = HopConfig(
        DMRSsymbols=torch.zeros((0,), dtype=torch.bool, device=device),
        DMRSREmask=torch.zeros((12, 0), dtype=torch.bool, device=device),
        PRBstart=0,
        nPRBs=0,
        maskPRBs=torch.zeros((0,), dtype=torch.bool, device=device),
        startSymbol=0,
        nAllocatedSymbols=0,
    )

    # Normal CP @ 15 kHz, FFT 2048 ⇒ CP samples [160, 144×13]
    Ts = 1.0 / (15000 * 2048)  # s
    cp_samples = np.array([160] + [144] * 13, dtype=np.float64)
    cp_ms = cp_samples * Ts * 1000.0

    config = EstimatorConfig(
        scs=15_000,
        CyclicPrefixDurations=torch.as_tensor(cp_ms, device=device),
        Smoothing="filter",
        CFOCompensate=True,
    )

    # Pilots reshape: order in file matches DMRS symbols ascending, subcarriers ascending.
    n_dmrs_symbols = 4
    n_re_per_sym = int(pilots.size // n_dmrs_symbols)
    n_layers = 1
    pilots_t = torch.as_tensor(
        pilots.reshape(n_dmrs_symbols, n_re_per_sym, n_layers).transpose(1, 0, 2),
        device=device,
    )

    # Full resource grid reconstruction from sparse entries.
    rg_np = entries_to_grid(rg_entries, n_sc_total=52 * 12, n_sym_total=14)
    ch_ref_np = entries_to_grid(ch_entries, n_sc_total=52 * 12, n_sym_total=14)

    # Convert to torch
    rg_t = torch.as_tensor(rg_np, device=device)
    ch_ref_t = torch.as_tensor(ch_ref_np, device=device)

    print(f"✅ Loaded: RG{rg_t.shape}, pilots{pilots_t.shape}, ref{ch_ref_t.shape}")

    # 네 함수 호출 (single layer라 마지막 dim 제거)
    rg_in = rg_t[:, :, 0] if rg_t.ndim == 3 and rg_t.shape[2] == 1 else rg_t

    with torch.no_grad():
        ch_torch, *_ = srs_channel_estimator(rg_in, pilots_t, 1.4125, hop1_config, hop2_config, config)

    diff = ch_torch - ch_ref_t
    max_err = torch.max(torch.abs(diff)).item()
    rms_err = torch.sqrt(torch.mean(torch.abs(diff) ** 2)).item()

    print(f"🎯 Max diff: {max_err:.2e}")
    print(f"📊 RMS diff: {rms_err:.2e}")


if __name__ == "__main__":
    main()
