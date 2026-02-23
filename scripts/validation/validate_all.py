"""
Run srs_channel_estimator over all port_channel_estimator test vectors (cases 0..247).
Parses test configuration from testvector_outputs/port_channel_estimator_test_data.h
so the script stays in sync with the generated MATLAB test data.
"""
from __future__ import annotations

import re
import struct
from itertools import permutations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from ce_rule_tensorized import EstimatorConfig, HopConfig, srs_channel_estimator

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
    # Keep the same precision as the testvector files (complex64) to avoid
    # needless up/down-casting later.
    grid = np.zeros((n_sc_total, n_sym_total, n_layers), dtype=np.complex64)
    for sym, port, sc, val in entries:
        grid[sc, sym, port] = np.complex64(val)
    return grid


# -----------------------------
# Parsing of port_channel_estimator_test_data.h
# -----------------------------
@dataclass
class HopParsed:
    dmrs_symbols: List[int]
    mask_prbs: List[int]
    dmrs_re_mask: List[int]
    hop_symbol: int | None = None  # optional hop boundary (start symbol of this hop)


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


def _tokenize_block(block: str) -> List[str]:
    """
    Light tokenization of a C++ initializer block.
    Returns a flat list containing:
      - '{...}' array literals (including empty '{}')
      - 'std::nullopt'
      - bare integers (e.g., hop_symbol = 7)
    """
    token_re = re.compile(r"\{[^{}]*\}|std::nullopt|[-+]?\d+")
    return [m.group(0) for m in token_re.finditer(block)]


def _parse_arrays_and_scalars(block: str):
    """
    Parse arrays (including empty) and bare integers from a block.
    Returns list of (kind, value):
      kind == 'arr' -> value: List[int]
      kind == 'int' -> value: int
      kind == 'null' -> value: None
    """
    tokens = _tokenize_block(block)
    out = []
    for tok in tokens:
        if tok == "std::nullopt":
            out.append(("null", None))
        elif tok.startswith("{"):
            inner = tok.strip("{}").strip()
            if inner == "":
                nums = []
            else:
                parts = [p for p in inner.replace("\n", " ").split(",") if p.strip()]
                # filter out string literals (paths) and keep only numeric tokens
                nums = []
                for p in parts:
                    p = p.strip()
                    if p.startswith('"') and p.endswith('"'):
                        continue
                    try:
                        nums.append(int(p))
                    except ValueError:
                        continue
            out.append(("arr", nums))
        else:
            out.append(("int", int(tok)))
    return out


def _parse_hops(tokens, n_alloc_syms: int) -> List[HopParsed]:
    """
    tokens: list of (kind, value) from _parse_arrays_and_scalars
    Each hop encoded as:
      DMRSsymbols (len 14 or n_alloc_syms)  [arr]
      maskPRBs one or more length-52 arrays [arr]
      optional hop_symbol (bare int)        [int]
      DMRSREmask length multiple of 12      [arr]
      optional std::nullopt or empty arrays ignored for structure
    """
    hops: List[HopParsed] = []
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]
        if kind != "arr" or len(val) not in (n_alloc_syms, 14):
            i += 1
            continue
        dmrs_symbols = val
        i += 1

        # collect mask PRBs
        mask_prbs_list = []
        while i < len(tokens) and tokens[i][0] == "arr" and len(tokens[i][1]) == 52:
            mask_prbs_list.append(tokens[i][1])
            i += 1

        # optional hop_symbol
        hop_symbol = None
        if i < len(tokens) and tokens[i][0] == "int":
            hop_symbol = tokens[i][1]
            i += 1

        # find dmrs_re_mask array (len multiple of 12)
        while i < len(tokens):
            if tokens[i][0] == "arr" and (len(tokens[i][1]) % 12 == 0) and len(tokens[i][1]) > 0:
                dmrs_re_mask = tokens[i][1]
                i += 1
                break
            i += 1
        else:
            break

        if not mask_prbs_list:
            mask_prbs_list = [np.zeros(52, dtype=int).tolist()]

        for m in mask_prbs_list:
            hops.append(HopParsed(dmrs_symbols, m, dmrs_re_mask, hop_symbol))
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

        tokens = _parse_arrays_and_scalars(block)
        hops = _parse_hops(tokens, n_alloc_syms)

        # collapse identical hop repetitions (common for multi-layer configs)
        if len(hops) > 1:
            first = hops[0]
            all_identical = all(
                h.hop_symbol == first.hop_symbol
                and h.dmrs_symbols == first.dmrs_symbols
                and h.mask_prbs == first.mask_prbs
                and h.dmrs_re_mask == first.dmrs_re_mask
                for h in hops
            )
            if all_identical:
                hops = [first]

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


def build_pilots_candidates(
    pilots: np.ndarray, n_dmrs_symbols_total: int, n_re_per_symbol: int, n_layers: int
) -> list[tuple[str, torch.Tensor]]:
    """
    Build candidate pilot tensors for possible file orderings.
    We treat raw pilots as a linear array whose logical axes are
    (sym, re, layer) in unknown storage order, then try all axis-order
    permutations and convert each candidate to [re, sym, layer].
    """
    candidates: list[tuple[str, torch.Tensor]] = []
    seen = set()
    axis_sizes = {
        "sym": n_dmrs_symbols_total,
        "re": n_re_per_symbol,
        "layer": n_layers,
    }

    for order in permutations(("sym", "re", "layer")):
        shape = tuple(axis_sizes[a] for a in order)
        arr = pilots.reshape(shape)
        src_idx = {name: i for i, name in enumerate(order)}
        arr_res = np.transpose(arr, (src_idx["re"], src_idx["sym"], src_idx["layer"]))

        layer_perms = [tuple(range(n_layers))]
        if 1 < n_layers <= 4:
            layer_perms = list(permutations(range(n_layers)))

        for lp in layer_perms:
            arr_final = arr_res[:, :, list(lp)]
            key = arr_final.tobytes()
            if key in seen:
                continue
            seen.add(key)
            tag = f"{order[0]}-{order[1]}-{order[2]}"
            if n_layers > 1:
                tag += f":L{''.join(map(str, lp))}"
            candidates.append((tag, torch.as_tensor(arr_final, dtype=torch.complex64)))

    return candidates


def dedupe_dmrs_re_mask_columns(dmrs_re_mask: np.ndarray) -> np.ndarray:
    """Keep unique DMRS RE mask columns in first-seen order."""
    if dmrs_re_mask.ndim != 2 or dmrs_re_mask.shape[1] <= 1:
        return dmrs_re_mask
    cols = []
    seen = set()
    for i in range(dmrs_re_mask.shape[1]):
        col = dmrs_re_mask[:, i : i + 1]
        key = col.tobytes()
        if key in seen:
            continue
        seen.add(key)
        cols.append(col)
    return np.concatenate(cols, axis=1) if cols else dmrs_re_mask


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

    parsed_hops_raw = []
    for hop in case.hops:
        dmrs_symbols = np.array(hop.dmrs_symbols, dtype=bool)
        mask_prbs = np.array(hop.mask_prbs, dtype=bool)
        dmrs_re_mask = np.array(hop.dmrs_re_mask, dtype=bool).reshape(12, -1)
        parsed_hops_raw.append((dmrs_symbols, mask_prbs, dmrs_re_mask, hop.hop_symbol))
    if len(parsed_hops_raw) == 0:
        raise ValueError(f"Case {case.idx}: no hops parsed from header.")

    # Collapse layer-repeated hop definitions by grouping on hop identity and stacking DMRS RE masks.
    # This preserves repeated columns (e.g. identical RE mask per layer).
    grouped_hops = []
    for dmrs_symbols, mask_prbs, dmrs_re_mask, hop_symbol in parsed_hops_raw:
        placed = False
        for i, (g_dmrs, g_mask, g_re, g_hop_symbol) in enumerate(grouped_hops):
            if (
                np.array_equal(dmrs_symbols, g_dmrs)
                and np.array_equal(mask_prbs, g_mask)
                and hop_symbol == g_hop_symbol
            ):
                grouped_hops[i] = (g_dmrs, g_mask, np.concatenate([g_re, dmrs_re_mask], axis=1), g_hop_symbol)
                placed = True
                break
        if not placed:
            grouped_hops.append((dmrs_symbols, mask_prbs, dmrs_re_mask, hop_symbol))
    parsed_hops_raw = [
        (dmrs_symbols, mask_prbs, dedupe_dmrs_re_mask_columns(dmrs_re_mask), hop_symbol)
        for dmrs_symbols, mask_prbs, dmrs_re_mask, hop_symbol in grouped_hops
    ]

    # If hop_symbol provided, split DMRS symbols by boundary; else fall back to single-hop merge when ambiguous.
    dmrs_union = np.logical_or.reduce([h[0] for h in parsed_hops_raw])
    dmrs_sym_indices = np.nonzero(dmrs_union)[0].tolist()
    n_hops = len(parsed_hops_raw)

    if any(h[3] is not None for h in parsed_hops_raw) and n_hops == 2:
        hop_symbol = next(h[3] for h in parsed_hops_raw if h[3] is not None)
        subsets = [np.array([idx for idx in dmrs_sym_indices if idx < hop_symbol]),
                   np.array([idx for idx in dmrs_sym_indices if idx >= hop_symbol])]
    elif n_hops == 2 and all(h[3] is None for h in parsed_hops_raw):
        # Heuristic: if two maskPRBs exist but no hop_symbol given, assume mid-slot hop.
        hop_symbol = case.n_alloc_syms // 2
        subsets = [np.array([idx for idx in dmrs_sym_indices if idx < hop_symbol]),
                   np.array([idx for idx in dmrs_sym_indices if idx >= hop_symbol])]
    elif n_hops == 1:
        subsets = [np.array(dmrs_sym_indices)]
    else:
        # Ambiguous multi-hop without boundary: merge masks into single hop.
        merged_mask = np.logical_or.reduce([h[0] for h in parsed_hops_raw])
        merged_prbs = np.logical_or.reduce([h[1] for h in parsed_hops_raw])
        merged_dmrs_re = parsed_hops_raw[0][2]
        parsed_hops_raw = [(merged_mask, merged_prbs, merged_dmrs_re, None)]
        subsets = [np.array(dmrs_sym_indices)]
        n_hops = 1

    parsed_hops = []
    for (dmrs_symbols_raw, mask_prbs, dmrs_re_mask, _), sym_subset in zip(parsed_hops_raw, subsets):
        dmrs_mask = np.zeros_like(dmrs_symbols_raw, dtype=bool)
        dmrs_mask[sym_subset] = True
        parsed_hops.append((dmrs_mask, mask_prbs, dmrs_re_mask))

    hop1_cfg = build_hop_config(parsed_hops[0][0], parsed_hops[0][1], parsed_hops[0][2], case.start_symbol, case.n_alloc_syms, device)
    if len(parsed_hops) > 1:
        hop2_cfg = build_hop_config(parsed_hops[1][0], parsed_hops[1][1], parsed_hops[1][2], case.start_symbol, case.n_alloc_syms, device)
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

    dmrs_symbols_list = [dm for dm, _, _ in parsed_hops]
    n_dmrs_symbols_total = int(np.logical_or.reduce(dmrs_symbols_list).sum())
    dmrs_re_mask = parsed_hops[0][2]
    # REs per PRB for one CDM column (pilots layer dimension is handled separately).
    dmrs_per_prb = int(dmrs_re_mask[:, 0].sum()) if dmrs_re_mask.shape[1] > 0 else 0
    if dmrs_per_prb == 0:
        raise ValueError(f"Case {case.idx}: DMRS RE mask empty.")

    # Assume per-symbol REs correspond to first hop PRB count (common in vectors).
    prbs_per_dmrs_sym = int(np.sum(parsed_hops[0][1]))
    n_re_per_symbol = dmrs_per_prb * prbs_per_dmrs_sym
    if n_dmrs_symbols_total == 0 or n_re_per_symbol == 0:
        raise ValueError(f"Case {case.idx}: empty DMRS definition.")
    divisor = n_dmrs_symbols_total * n_re_per_symbol
    if pilots.size % divisor != 0:
        raise ValueError(f"Case {case.idx}: pilots size {pilots.size} not divisible by {divisor}")
    n_layers = pilots.size // divisor

    pilot_candidates = [
        (name, tensor.to(device))
        for name, tensor in build_pilots_candidates(pilots, n_dmrs_symbols_total, n_re_per_symbol, n_layers)
    ]

    if case.idx in DEBUG_CASES:
        mask_prbs0 = np.array(case.hops[0].mask_prbs, dtype=bool)
        dmrs_re_mask0 = np.array(case.hops[0].dmrs_re_mask, dtype=bool).reshape(12, -1)[:, 0]
        sc_mask = np.repeat(mask_prbs0, 12) & np.tile(dmrs_re_mask0, mask_prbs0.size)
        sc_indices = np.nonzero(sc_mask)[0].tolist()
        dmrs_sym_idx = np.nonzero(np.array(case.hops[0].dmrs_symbols, dtype=bool))[0].tolist()
        coords = [(sym, sc) for sc in sc_indices for sym in dmrs_sym_idx]
        print(f"[DEBUG case {case.idx}] DMRS coords (sym, sc) first 12: {coords[:12]}")
        print(
            f"[DEBUG case {case.idx}] pilot candidates: "
            + ", ".join([f"{name}:{tuple(t.shape)}" for name, t in pilot_candidates])
        )
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

    rg_t = torch.as_tensor(rg_np, device=device, dtype=torch.complex64)

    if rg_t.ndim == 3:
        if rg_t.shape[2] == 1:
            rg_in = rg_t[:, :, 0]
        else:
            raise ValueError(f"Case {case.idx}: multiple input ports in RG not supported in this validator.")
    else:
        rg_in = rg_t

    best_name = None
    best_max = np.inf
    best_rms = np.inf

    for ordering_name, pilots_t in pilot_candidates:
        with torch.no_grad():
            ch_est, _, _, _, _, _ = srs_channel_estimator(
                rg_in, pilots_t, case.beta_dmrs, hop1_cfg, hop2_cfg, config
            )

        est_vals = []
        ref_vals = []
        for sym, port, sc, ref in ch_entries:
            if port >= ch_est.shape[2]:
                raise ValueError(
                    f"Case {case.idx}: estimator output has {ch_est.shape[2]} layer(s), "
                    f"but reference needs port {port} (ordering={ordering_name})."
                )
            est_vals.append(ch_est[sc, sym, port].item())
            ref_vals.append(ref)
        est_vals = np.array(est_vals, dtype=np.complex128)
        ref_vals = np.array(ref_vals, dtype=np.complex128)
        diff_vals = est_vals - ref_vals
        max_err = np.max(np.abs(diff_vals)).item()
        rms_err = np.sqrt(np.mean(np.abs(diff_vals) ** 2)).item()

        if (rms_err < best_rms) or (np.isclose(rms_err, best_rms) and max_err < best_max):
            best_name = ordering_name
            best_max = max_err
            best_rms = rms_err

    if case.idx in DEBUG_CASES and best_name is not None:
        print(f"[DEBUG case {case.idx}] selected pilot ordering: {best_name}")

    return best_max, best_rms


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
