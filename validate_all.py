"""
Run srs_channel_estimator over all port_channel_estimator test vectors (cases 0..247).
Parses test configuration from testvector_outputs/port_channel_estimator_test_data.h
so the script stays in sync with the generated MATLAB test data.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from srs_estimator_torch import EstimatorConfig, HopConfig, srs_channel_estimator

HEADER_PATH = Path("testvector_outputs/port_channel_estimator_test_data.h")
DATA_DIR = Path("testvector_outputs")
DEBUG_CASES = {4}  # set of case indices to dump DMRS coordinate mapping


# -----------------------------
# I/O helpers (copied from validate_case0 with minor generalizations)
# -----------------------------
def load_expected_entries(path: Path):
    """Parse resource_grid_reader_spy::expected_entry_t dumped by file_vector."""
    raw = path.read_bytes()
    if len(raw) % 12 != 0:
        raise ValueError(f"{path} size {len(raw)} not multiple of 12 bytes.")
    out = []
    for i in range(0, len(raw), 12):
        packed, sc, re_, im_ = struct.unpack_from("<HHff", raw, i)
        sym = packed >> 8
        port = packed & 0xFF
        out.append((sym, port, sc, re_ + 1j * im_))
    return out


def entries_to_grid(entries, n_sc_total: int, n_sym_total: int):
    n_layers = max(p for _, p, _, _ in entries) + 1
    grid = np.zeros((n_sc_total, n_sym_total, n_layers), dtype=np.complex128)
    for sym, port, sc, val in entries:
        grid[sc, sym, port] = val
    return grid


# -----------------------------
# Parsing of port_channel_estimator_test_data.h
# -----------------------------
@dataclass
class HopParsed:
    dmrs_symbols: List[int]
    mask_prbs: List[int]
    dmrs_re_mask: List[int]


@dataclass
class TestCaseParsed:
    idx: int
    scs_hz: float
    start_symbol: int
    n_alloc_syms: int
    beta_dmrs: float
    smoothing: str
    cfo_compensate: bool
    grid_size_prbs: int
    hops: List[HopParsed]


def _split_top_level_cases(text: str) -> List[str]:
    marker = "port_channel_estimator_test_data"
    pos = text.index(marker)
    brace_start = text.index("{", text.index("=", pos))
    cases = []
    depth = 0
    in_str = False
    start = None
    for i, ch in enumerate(text[brace_start:], start=brace_start):
        if ch == '"' and text[i - 1] != "\\":
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "}":
            if depth == 2 and start is not None:
                cases.append(text[start : i + 1])
                start = None
            depth -= 1
        if depth == 0 and cases:
            break
    return cases


def _parse_arrays(block: str) -> List[List[int]]:
    arr_strings = [a for a in re.findall(r"\{([0-9,\s]+)\}", block) if a.strip()]
    arrays = []
    for s in arr_strings:
        nums = [int(x) for x in s.replace("\n", " ").split(",") if x.strip()]
        arrays.append(nums)
    return arrays


def _parse_hops(arrays: List[List[int]], n_alloc_syms: int) -> List[HopParsed]:
    """
    Each hop in the header encodes (in order):
      - DMRSsymbols : length 14 (slot) or n_alloc_syms
      - maskPRBs    : length 52 (full grid)
      - DMRSREmask  : length multiple of 12 (flattened 12 x n_cdm)
    This repeats for hop2, hop3... if present.
    Empty braces and std::nullopt are ignored by _parse_arrays.
    """
    hops: List[HopParsed] = []
    i = 0
    while i < len(arrays):
        dmrs_len_ok = len(arrays[i]) in (n_alloc_syms, 14)
        if not dmrs_len_ok:
            i += 1
            continue
        dmrs_symbols = arrays[i]
        i += 1
        if i >= len(arrays):
            break

        # maskPRBs should have length 52 (one per PRB position in 20 MHz grid)
        mask_prbs = arrays[i]
        if len(mask_prbs) != 52:
            # If unexpected length, skip this potential hop.
            continue
        i += 1
        if i >= len(arrays):
            break

        # Find next array whose length is a multiple of 12 => DMRS RE mask
        while i < len(arrays) and (len(arrays[i]) % 12 != 0):
            i += 1
        if i >= len(arrays):
            break
        dmrs_re_mask = arrays[i]
        i += 1

        hops.append(HopParsed(dmrs_symbols, mask_prbs, dmrs_re_mask))
    return hops


def parse_header() -> List[TestCaseParsed]:
    text = HEADER_PATH.read_text()
    blocks = _split_top_level_cases(text)
    cases: List[TestCaseParsed] = []
    for block in blocks:
        idx_match = re.search(r"input_rg(\d+)", block)
        idx = int(idx_match.group(1)) if idx_match else len(cases)

        scs_match = re.search(r"subcarrier_spacing::kHz(\d+)", block)
        scs_khz = int(scs_match.group(1)) if scs_match else 15
        scs_hz = scs_khz * 1000.0

        start_sym_match = re.search(r"cyclic_prefix::\w+,\s*(\d+),\s*(\d+)", block)
        start_symbol, n_alloc_syms = (0, 14)
        if start_sym_match:
            start_symbol = int(start_sym_match.group(1))
            n_alloc_syms = int(start_sym_match.group(2))

        smooth_match = re.search(r"port_channel_estimator_fd_smoothing_strategy::(\w+)", block)
        smoothing = smooth_match.group(1) if smooth_match else "filter"

        cfo_match = re.search(r"port_channel_estimator_fd_smoothing_strategy::\w+,\s*(true|false)", block)
        cfo_compensate = cfo_match.group(1) == "true" if cfo_match else True

        grid_match = re.search(r"port_channel_estimator_fd_smoothing_strategy::\w+,\s*(?:true|false),\s*(\d+)", block)
        grid_size_prbs = int(grid_match.group(1)) if grid_match else 52

        # Beta DMRS: last float before smoothing enum
        smooth_pos = smooth_match.start() if smooth_match else 0
        prefix = block[:smooth_pos]
        float_candidates = re.findall(r"[-+]?[0-9]*\.?[0-9]+", prefix)
        beta_dmrs = float(float_candidates[-1]) if float_candidates else 1.4125

        arrays = _parse_arrays(block)
        hops = _parse_hops(arrays, n_alloc_syms)

        cases.append(
            TestCaseParsed(
                idx=idx,
                scs_hz=scs_hz,
                start_symbol=start_symbol,
                n_alloc_syms=n_alloc_syms,
                beta_dmrs=beta_dmrs,
                smoothing=smoothing,
                cfo_compensate=cfo_compensate,
                grid_size_prbs=grid_size_prbs,
                hops=hops,
            )
        )
    cases.sort(key=lambda c: c.idx)
    return cases


# -----------------------------
# Utility functions
# -----------------------------
def compute_cp_ms(scs_hz: float, n_syms: int = 14) -> np.ndarray:
    """Return CP durations in ms for a normal CP slot at given SCS.

    We scale the 15 kHz reference lengths (160, 144) by 15k / scs to keep
    sampling rate consistent with the 20 MHz LTE numerology used by srsRAN.
    """
    scale = 15000.0 / scs_hz
    cp0 = int(round(160 * scale))
    cp_rest = int(round(144 * scale))
    cp_samples = np.array([cp0] + [cp_rest] * (n_syms - 1), dtype=np.float64)

    # Keep FFT size at 2048 (NR/LTE common base) and adjust timing only via SCS.
    n_fft = 2048
    Ts = 1.0 / (scs_hz * n_fft)
    return cp_samples * Ts * 1000.0


def build_hop_config(
    dmrs_symbols: np.ndarray, mask_prbs: np.ndarray, dmrs_re_mask: np.ndarray, start_symbol: int, n_alloc_syms: int, device: str
) -> HopConfig:
    dmrs_symbols_t = torch.as_tensor(dmrs_symbols, dtype=torch.bool, device=device)
    mask_prbs_t = torch.as_tensor(mask_prbs, dtype=torch.bool, device=device)
    n_prbs = int(mask_prbs_t.to(torch.int64).sum().item())
    prb_start = int(torch.nonzero(mask_prbs_t, as_tuple=False)[0].item()) if n_prbs > 0 else 0
    dmrs_re_mask_t = torch.as_tensor(dmrs_re_mask, dtype=torch.bool, device=device).view(12, -1)

    return HopConfig(
        DMRSsymbols=dmrs_symbols_t,
        DMRSREmask=dmrs_re_mask_t,
        PRBstart=prb_start,
        nPRBs=n_prbs,
        maskPRBs=mask_prbs_t,
        startSymbol=start_symbol,
        nAllocatedSymbols=n_alloc_syms,
    )


# -----------------------------
# Main runner
# -----------------------------
def run_case(case: TestCaseParsed, device: str = "cpu"):
    input_path = DATA_DIR / f"port_channel_estimator_test_input_rg{case.idx}.dat"
    pilots_path = DATA_DIR / f"port_channel_estimator_test_pilots{case.idx}.dat"
    output_path = DATA_DIR / f"port_channel_estimator_test_output_ch_est{case.idx}.dat"

    rg_entries = load_expected_entries(input_path)
    ch_entries = load_expected_entries(output_path)
    pilots = np.fromfile(pilots_path, dtype=np.complex64)

    n_sc_total = case.grid_size_prbs * 12
    max_sym_rg = max(sym for sym, _, _, _ in rg_entries)
    max_sym_ch = max(sym for sym, _, _, _ in ch_entries)
    max_sym_idx = max(max_sym_rg, max_sym_ch)
    n_sym_total = max(case.n_alloc_syms, max_sym_idx + 1, 14)

    rg_np = entries_to_grid(rg_entries, n_sc_total=n_sc_total, n_sym_total=n_sym_total)
    ch_ref_np = entries_to_grid(ch_entries, n_sc_total=n_sc_total, n_sym_total=n_sym_total)

    parsed_hops = []
    for hop in case.hops:
        dmrs_symbols = np.array(hop.dmrs_symbols, dtype=bool)
        mask_prbs = np.array(hop.mask_prbs, dtype=bool)
        dmrs_re_mask = np.array(hop.dmrs_re_mask, dtype=bool).reshape(12, -1)
        parsed_hops.append((dmrs_symbols, mask_prbs, dmrs_re_mask))
    if len(parsed_hops) == 0:
        raise ValueError(f"Case {case.idx}: no hops parsed from header.")

    hop1_cfg = build_hop_config(parsed_hops[0][0], parsed_hops[0][1], parsed_hops[0][2], case.start_symbol, case.n_alloc_syms, device)
    if len(parsed_hops) > 1:
        dmrs2 = parsed_hops[1][0] & (~parsed_hops[0][0])
        hop2_cfg = build_hop_config(dmrs2, parsed_hops[1][1], parsed_hops[1][2], case.start_symbol, case.n_alloc_syms, device)
    else:
        hop2_cfg = HopConfig(
            DMRSsymbols=torch.zeros((0,), dtype=torch.bool, device=device),
            DMRSREmask=torch.zeros((12, 0), dtype=torch.bool, device=device),
            PRBstart=0,
            nPRBs=0,
            maskPRBs=torch.zeros((0,), dtype=torch.bool, device=device),
            startSymbol=0,
            nAllocatedSymbols=0,
        )

    cp_ms = compute_cp_ms(case.scs_hz, n_syms=14)
    config = EstimatorConfig(
        scs=case.scs_hz,
        CyclicPrefixDurations=torch.as_tensor(cp_ms, device=device),
        Smoothing=case.smoothing,
        CFOCompensate=case.cfo_compensate,
    )

    dmrs_symbols_list = [np.array(h.dmrs_symbols, dtype=bool) for h in case.hops]
    n_dmrs_symbols_total = int(sum(int(dm.sum()) for dm in dmrs_symbols_list))
    dmrs_re_mask = np.array(case.hops[0].dmrs_re_mask, dtype=bool).reshape(12, -1)
    dmrs_per_prb = int(dmrs_re_mask[:, 0].sum())
    n_prbs = int(np.sum(case.hops[0].mask_prbs))
    n_re_per_symbol = n_prbs * dmrs_per_prb
    if n_dmrs_symbols_total == 0 or n_re_per_symbol == 0:
        raise ValueError(f"Case {case.idx}: empty DMRS definition")
    n_layers = int(pilots.size // (n_dmrs_symbols_total * n_re_per_symbol))
    if n_layers == 0:
        raise ValueError(f"Case {case.idx}: could not infer n_layers from pilots")

    # Normalize pilots by beta_dmrs to avoid double scaling.
    pilots_norm = pilots / case.beta_dmrs
    pilots_t = torch.as_tensor(
        pilots_norm.reshape(n_dmrs_symbols_total, n_re_per_symbol, n_layers).transpose(1, 0, 2),
        device=device,
    )

    if case.idx in DEBUG_CASES:
        mask_prbs0 = np.array(case.hops[0].mask_prbs, dtype=bool)
        dmrs_re_mask0 = np.array(case.hops[0].dmrs_re_mask, dtype=bool).reshape(12, -1)[:, 0]
        sc_mask = np.repeat(mask_prbs0, 12) & np.tile(dmrs_re_mask0, mask_prbs0.size)
        sc_indices = np.nonzero(sc_mask)[0].tolist()
        dmrs_sym_idx = np.nonzero(np.array(case.hops[0].dmrs_symbols, dtype=bool))[0].tolist()
        coords = [(sym, sc) for sc in sc_indices for sym in dmrs_sym_idx]
        print(f"[DEBUG case {case.idx}] DMRS coords (sym, sc) first 12: {coords[:12]}")
        print(f"[DEBUG case {case.idx}] pilots_t shape {pilots_t.shape}, n_layers {n_layers}")
        rg_debug = torch.as_tensor(rg_np, device=device)
        if rg_debug.ndim == 3:
            rg_debug = rg_debug.sum(dim=2)
        vals = []
        for sc in sc_indices:
            for sym in dmrs_sym_idx:
                vals.append(rg_debug[sc, sym].item())
        vals = np.array(vals, dtype=np.complex128)
        pilots_sc_major = pilots.reshape(n_dmrs_symbols_total, n_re_per_symbol).T.flatten()
        # quick channel-from-DMRS estimate at pilot REs
        h_est_dmrs = vals / pilots_sc_major[: vals.size]
        # reference channel at same coords
        ref_vals = []
        for sc in sc_indices:
            for sym in dmrs_sym_idx:
                ref_vals.append(ch_ref_np[sc, sym, 0])
        ref_vals = np.array(ref_vals, dtype=np.complex128)
        # optimal complex scalar alignment
        num = np.vdot(ref_vals, h_est_dmrs)
        den = np.vdot(ref_vals, ref_vals)
        alpha = num / den if den != 0 else 1.0
        aligned = h_est_dmrs / alpha
        diff_dbg = aligned - ref_vals
        print(f"[DEBUG case {case.idx}] DMRS h_est vs ref (after scalar align): max {np.max(np.abs(diff_dbg)):.3e}, rms {np.sqrt(np.mean(np.abs(diff_dbg)**2)):.3e}")

    rg_t = torch.as_tensor(rg_np, device=device)
    ch_ref_t = torch.as_tensor(ch_ref_np, device=device)

    if rg_t.ndim == 3:
        rg_in = rg_t.sum(dim=2)
    else:
        rg_in = rg_t

    with torch.no_grad():
        ch_est, _, _, _, _, _ = srs_channel_estimator(
            rg_in, pilots_t, case.beta_dmrs, hop1_cfg, hop2_cfg, config
        )

    diff = ch_est - ch_ref_t
    max_err = torch.max(torch.abs(diff)).item()
    rms_err = torch.sqrt(torch.mean(torch.abs(diff) ** 2)).item()

    return max_err, rms_err


def main():
    cases = parse_header()
    print(f"Discovered {len(cases)} test cases (expected 248).")

    worst_max = 0.0
    worst_case = None
    for case in cases:
        max_err, rms_err = run_case(case)
        print(f"Case {case.idx:3d}: max {max_err:.2e}  rms {rms_err:.2e}")
        if max_err > worst_max:
            worst_max = max_err
            worst_case = (case.idx, rms_err)

    if worst_case:
        idx, rms = worst_case
        print(f"Worst case {idx}: max {worst_max:.2e}, rms {rms:.2e}")


if __name__ == "__main__":
    main()
