"""
Validate `srs_estimator_torch.srs_channel_estimator` against test case 4
from the port_channel_estimator reference vectors.

Case 4 exercises two frequency hops (PRBs 3–5 then 28–30) with hop at
symbol 7. We mirror the header exactly instead of “half split”.
"""

import struct
from pathlib import Path

import numpy as np
import torch

from srs_estimator_torch import EstimatorConfig, HopConfig, srs_channel_estimator


def load_expected_entries(path: str):
    """
    Parse resource_grid_reader_spy::expected_entry_t dumped by file_vector.
    Layout (little endian, 12 bytes per entry):
      uint16 packed_sym_port  (symbol = packed >> 8, port = packed & 0xFF)
      uint16 subcarrier_index (absolute)
      float32 real
      float32 imag
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


# -----------------------------
# Case 4 data paths
# -----------------------------
rg_entries = load_expected_entries("./testvector_outputs/port_channel_estimator_test_input_rg4.dat")
ch_entries = load_expected_entries("./testvector_outputs/port_channel_estimator_test_output_ch_est4.dat")
pilots = np.fromfile("./testvector_outputs/port_channel_estimator_test_pilots4.dat", dtype=np.complex64)


# -----------------------------
# Config for case 4 (parsed from port_channel_estimator_test_data.h)
# -----------------------------
dmrs_symbols_union = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)

# Hop boundary from header (hop_symbol = 7)
hop_symbol = 7
dmrs_symbols_h1 = dmrs_symbols_union & (np.arange(dmrs_symbols_union.size) < hop_symbol)
dmrs_symbols_h2 = dmrs_symbols_union & (np.arange(dmrs_symbols_union.size) >= hop_symbol)

# PRB masks: 3 PRBs at indices 3–5, and 3 PRBs at indices 28–30.
mask_prbs_h1 = np.zeros(52, dtype=bool)
mask_prbs_h1[[3, 4, 5]] = True

mask_prbs_h2 = np.zeros(52, dtype=bool)
mask_prbs_h2[[28, 29, 30]] = True

# DMRS RE mask (12 REs per PRB, CDM=1) pattern 1010...
dmrs_re_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool).reshape(12, 1)


def main():
    device = "cpu"

    hop1_config = HopConfig(
        DMRSsymbols=torch.as_tensor(dmrs_symbols_h1, device=device),
        DMRSREmask=torch.as_tensor(dmrs_re_mask, device=device),
        PRBstart=3,
        nPRBs=int(mask_prbs_h1.sum()),
        maskPRBs=torch.as_tensor(mask_prbs_h1, device=device),
        startSymbol=0,
        nAllocatedSymbols=14,
    )

    hop2_config = HopConfig(
        DMRSsymbols=torch.as_tensor(dmrs_symbols_h2, device=device),
        DMRSREmask=torch.as_tensor(dmrs_re_mask, device=device),
        PRBstart=28,
        nPRBs=int(mask_prbs_h2.sum()),
        maskPRBs=torch.as_tensor(mask_prbs_h2, device=device),
        startSymbol=0,
        nAllocatedSymbols=14,
    )

    # Normal CP @ 15 kHz, FFT 2048 ⇒ CP samples [160, 144×13]
    Ts = 1.0 / (15000 * 2048)  # s
    cp_samples = np.array([160] + [144] * 13, dtype=np.float64)
    cp_ms = cp_samples * Ts * 1000.0

    config = EstimatorConfig(
        scs=15_000.0,
        CyclicPrefixDurations=torch.as_tensor(cp_ms, device=device),
        Smoothing="filter",
        CFOCompensate=True,
    )

    # Determine pilots layout from file size (no assumptions about PRB count).
    dmrs_per_prb = int(dmrs_re_mask.sum())  # 6 REs per PRB
    n_dmrs_symbols_total = int(dmrs_symbols_union.sum())  # 4
    # pilots.size = n_dmrs_symbols_total * n_prbs_per_sym * dmrs_per_prb * n_layers
    n_prbs_per_sym = pilots.size // (n_dmrs_symbols_total * dmrs_per_prb)
    if n_prbs_per_sym == 0:
        raise ValueError("Could not infer PRB count per DMRS symbol from pilots file.")
    n_layers = pilots.size // (n_dmrs_symbols_total * dmrs_per_prb * n_prbs_per_sym)
    if pilots.size != n_dmrs_symbols_total * dmrs_per_prb * n_prbs_per_sym * n_layers:
        raise ValueError("Pilots file size not divisible by inferred dims.")
    # Cross-check against header: hop1/2 each have 3 PRBs.
    expected_prbs = int(mask_prbs_h1.sum())
    assert n_prbs_per_sym == expected_prbs, f"Pilots PRB count {n_prbs_per_sym} != header {expected_prbs}"
    n_re_per_sym = dmrs_per_prb * n_prbs_per_sym

    pilots_t = torch.as_tensor(
        pilots.reshape(n_dmrs_symbols_total, n_re_per_sym, n_layers).transpose(1, 0, 2),
        device=device,
    )

    # Full resource grid reconstruction from sparse entries.
    n_sc_total = 52 * 12
    n_sym_total = 14
    rg_np = entries_to_grid(rg_entries, n_sc_total=n_sc_total, n_sym_total=n_sym_total)
    ch_ref_np = entries_to_grid(ch_entries, n_sc_total=n_sc_total, n_sym_total=n_sym_total)

    # Convert to torch
    rg_t = torch.as_tensor(rg_np, device=device)
    ch_ref_t = torch.as_tensor(ch_ref_np, device=device)

    print(f"✅ Loaded: RG{rg_t.shape}, pilots{pilots_t.shape}, ref{ch_ref_t.shape}")

    # Collapse layer dimension if single-layer input
    rg_in = rg_t[:, :, 0] if rg_t.ndim == 3 and rg_t.shape[2] == 1 else rg_t

    with torch.no_grad():
        ch_torch, *_ = srs_channel_estimator(rg_in, pilots_t, 1.4125, hop1_config, hop2_config, config)

    # Align complex gain using DMRS REs from hop1 for a fair comparison.
    mask_prbs0 = mask_prbs_h1
    dmrs_re_mask0 = dmrs_re_mask[:, 0]
    sc_mask = np.repeat(mask_prbs0, 12) & np.tile(dmrs_re_mask0, mask_prbs0.size)
    sc_indices = np.nonzero(sc_mask)[0].tolist()
    dmrs_sym_idx = np.nonzero(dmrs_symbols_union)[0].tolist()

    est_vals = []
    ref_vals = []
    for sc in sc_indices:
        for sym in dmrs_sym_idx:
            est_vals.append(ch_torch[sc, sym, 0].item())
            ref_vals.append(ch_ref_np[sc, sym, 0])
    est_vals = np.array(est_vals, dtype=np.complex128)
    ref_vals = np.array(ref_vals, dtype=np.complex128)

    # Compare only at reference entries to avoid penalising non-dumped REs.
    diff_vals = []
    for sym, port, sc, ref in ch_entries:
        est = ch_torch[sc, sym, port].item()
        diff_vals.append(est - ref)
    diff_vals = np.array(diff_vals, dtype=np.complex128)
    max_err = np.max(np.abs(diff_vals)).item()
    rms_err = np.sqrt(np.mean(np.abs(diff_vals) ** 2)).item()

    print(f"🎯 Max diff (ref entries only): {max_err:.2e}")
    print(f"📊 RMS diff (ref entries only): {rms_err:.2e}")


if __name__ == "__main__":
    main()
