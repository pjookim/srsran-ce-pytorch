"""
Diagnose whether `srs_estimator_torch.srs_channel_estimator` can be captured by
`torch.compile` for a given backend (e.g., furiosa), and report graph breaks.

Usage:
  .venv/bin/python diagnose_furiosa_graphbreak.py --backend furiosa
  .venv/bin/python diagnose_furiosa_graphbreak.py --backend inductor --fullgraph
"""

from __future__ import annotations

import argparse
import struct
import traceback
from pathlib import Path

import numpy as np
import torch
import torch._dynamo as dynamo

from srs_estimator_torch import EstimatorConfig, HopConfig, srs_channel_estimator


def load_expected_entries(path: str):
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
    grid = np.zeros((n_sc_total, n_sym_total, n_layers), dtype=np.complex64)
    for sym, port, sc, val in entries:
        grid[sc, sym, port] = np.complex64(val)
    return grid


def build_case8_inputs(device: str = "cpu"):
    rg_entries = load_expected_entries("./testvector_outputs/port_channel_estimator_test_input_rg8.dat")
    pilots = np.fromfile("./testvector_outputs/port_channel_estimator_test_pilots8.dat", dtype=np.complex64)

    dmrs_symbols = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    dmrs_re_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool).reshape(12, 1)
    mask_prbs = np.array(
        [
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        dtype=bool,
    )

    hop1_config = HopConfig(
        DMRSsymbols=torch.as_tensor(dmrs_symbols, device=device),
        DMRSREmask=torch.as_tensor(dmrs_re_mask, device=device),
        PRBstart=1,
        nPRBs=int(mask_prbs.sum()),
        maskPRBs=torch.as_tensor(mask_prbs, device=device),
        startSymbol=0,
        nAllocatedSymbols=14,
    )

    hop2_config = HopConfig(
        DMRSsymbols=torch.zeros((0,), dtype=torch.bool, device=device),
        DMRSREmask=torch.zeros((12, 0), dtype=torch.bool, device=device),
        PRBstart=0,
        nPRBs=0,
        maskPRBs=torch.zeros((0,), dtype=torch.bool, device=device),
        startSymbol=0,
        nAllocatedSymbols=0,
    )

    # Normal CP @ 15kHz
    Ts = 1.0 / (15000 * 2048)
    cp_samples = np.array([160] + [144] * 13, dtype=np.float64)
    cp_ms = cp_samples * Ts * 1000.0

    config = EstimatorConfig(
        scs=15_000,
        CyclicPrefixDurations=torch.as_tensor(cp_ms, device=device),
        Smoothing="filter",
        CFOCompensate=True,
    )

    n_dmrs_symbols = int(dmrs_symbols.sum())
    n_re_per_sym = int(mask_prbs.sum() * dmrs_re_mask.sum())
    denom = n_dmrs_symbols * n_re_per_sym
    if pilots.size % denom != 0:
        raise ValueError(f"Unexpected pilot size {pilots.size} (denom={denom}).")
    n_layers = pilots.size // denom

    pilots_t = torch.as_tensor(
        pilots.reshape(n_layers, n_dmrs_symbols, n_re_per_sym).transpose(2, 1, 0),
        device=device,
    )
    rg_np = entries_to_grid(rg_entries, n_sc_total=52 * 12, n_sym_total=14)
    rg_t = torch.as_tensor(rg_np, device=device)

    if rg_t.ndim == 3 and rg_t.shape[2] == 1:
        rg_in = rg_t[:, :, 0]
    else:
        raise ValueError(f"Expected single input port for case8, got shape {tuple(rg_t.shape)}")

    return rg_in, pilots_t, hop1_config, hop2_config, config


def format_break_reason(reason_obj):
    reason = getattr(reason_obj, "reason", "<unknown>")
    user_stack = getattr(reason_obj, "user_stack", None) or []
    stack_text = []
    for fr in user_stack:
        stack_text.append(f"{fr.filename}:{fr.lineno} in {fr.name}")
    return reason, stack_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="furiosa", help="torch.compile backend name")
    parser.add_argument("--fullgraph", action="store_true", help="Use fullgraph=True for strict capture")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic=True in torch.compile")
    args = parser.parse_args()

    device = "cpu"
    rg_in, pilots_t, hop1, hop2, cfg = build_case8_inputs(device=device)

    def model_fn(rg, pils):
        out = srs_channel_estimator(rg, pils, 1.4125, hop1, hop2, cfg)
        return out[0]

    print("== Environment ==")
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    print(f"requested_backend={args.backend}")
    try:
        backends = sorted(dynamo.list_backends())
    except Exception:
        backends = []
    print(f"available_backends={backends}")

    with torch.no_grad():
        eager_out = model_fn(rg_in, pilots_t)
    print(f"eager output shape={tuple(eager_out.shape)}, dtype={eager_out.dtype}")

    print("\n== torch.compile check ==")
    compile_ok = False
    try:
        compiled = torch.compile(
            model_fn,
            backend=args.backend,
            fullgraph=args.fullgraph,
            dynamic=args.dynamic,
        )
        with torch.no_grad():
            compiled_out = compiled(rg_in, pilots_t)
        max_diff = (compiled_out - eager_out).abs().max().item()
        print(f"compile: SUCCESS (max|compiled-eager|={max_diff:.3e})")
        compile_ok = True
    except Exception as e:
        print(f"compile: FAILED ({type(e).__name__}: {e})")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb)

    print("\n== Dynamo graph-break report ==")
    dynamo.reset()
    explain_out = dynamo.explain(model_fn)(rg_in, pilots_t)

    print(f"graph_count={explain_out.graph_count}")
    print(f"graph_break_count={explain_out.graph_break_count}")
    print(f"op_count={explain_out.op_count}")
    print(f"ops_per_graph={[len(ops) for ops in explain_out.ops_per_graph]}")

    if explain_out.break_reasons:
        for i, br in enumerate(explain_out.break_reasons, start=1):
            reason, stack = format_break_reason(br)
            print(f"[{i}] reason={reason}")
            if stack:
                for s in stack:
                    print(f"    at {s}")
            else:
                print("    at <no user stack>")
    else:
        print("No graph breaks reported by Dynamo for this run.")

    print("\n== NPU offload estimate (Dynamo-level) ==")
    if explain_out.graph_count <= 1 and explain_out.graph_break_count == 0:
        print("single-graph capture 가능: 백엔드가 해당 연산을 지원하면 대부분 NPU 오프로딩 가능")
    else:
        print("multi-graph capture: graph break 지점은 CPU fallback 가능성이 큼")
        print("break 원인/스택을 기준으로 코드 리팩터링 우선순위를 정하세요")

    if not compile_ok and args.backend not in backends:
        print(f"\nNOTE: backend '{args.backend}' 는 현재 환경에서 등록되어 있지 않습니다.")


if __name__ == "__main__":
    main()
