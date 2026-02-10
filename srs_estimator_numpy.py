import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict


# -----------------------------
# Data structures (MATLAB structs)
# -----------------------------
@dataclass
class Hop:
    # DMRSsymbols: logical mask over slot symbols, shape (n_sym_total,)
    DMRSsymbols: np.ndarray
    # DMRSREmask: logical mask within a PRB, shape (12, n_cdm)
    DMRSREmask: np.ndarray
    # PRBstart: first PRB dedicated to DMRS (0-based)
    PRBstart: int
    # nPRBs: number of PRBs dedicated to DMRS
    nPRBs: int
    # maskPRBs: logical mask for PRBs dedicated to DMRS (used with kron), shape (n_prb_mask_len,)
    maskPRBs: np.ndarray
    # startSymbol: hop start symbol index (0-based)
    startSymbol: int
    # nAllocatedSymbols: number of allocated symbols for hop
    nAllocatedSymbols: int


@dataclass
class Config:
    # subcarrier spacing in hertz
    scs: float
    # CP durations in milliseconds (as described in header comment) - MATLAB later scales by scs/1000
    CyclicPrefixDurations: np.ndarray
    # optional fields
    Smoothing: Optional[str] = None
    CFOCompensate: Optional[bool] = None


# -----------------------------
# Helper ports (already discussed earlier) - kept here so this file runs standalone.
# -----------------------------
def create_virtual_pilots(in_pilots: np.ndarray, n_virtuals: int) -> np.ndarray:
    """
    in_pilots: vector-like complex, length n_pilots
    returns virtual_pilots shape: (n_virtuals,)
    """
    in_pilots = np.asarray(in_pilots).reshape(-1)
    if n_virtuals < 0:
        raise ValueError("n_virtuals must be >= 0")
    if n_virtuals == 0:
        return np.empty((0,), dtype=np.complex128)

    n_pilots = in_pilots.size
    x = np.arange(n_pilots, dtype=np.float64)  # (n_pilots,)
    mx = x.mean()
    normx = np.linalg.norm(x) ** 2
    denom = (normx - n_pilots * (mx ** 2))

    v_idx = np.arange(-n_virtuals, 0, dtype=np.float64)  # (-nVirtuals:-1)' -> (n_virtuals,)

    # magnitude line fit
    y = np.abs(in_pilots).astype(np.float64)
    my = y.mean()
    a = (np.dot(x, y) - n_pilots * mx * my) / denom
    b = my - a * mx
    virtual = a * v_idx + b  # (n_virtuals,)

    # phase line fit
    y = np.unwrap(np.angle(in_pilots).astype(np.float64))
    my = y.mean()
    a = (np.dot(x, y) - n_pilots * mx * my) / denom
    b = my - a * mx
    ph = a * v_idx + b
    return virtual * np.exp(1j * ph)


def _rcosdesign_normal(beta: float, span: int, sps: int) -> np.ndarray:
    """NumPy-only equivalent of MATLAB rcosdesign(beta, span, sps, 'normal')."""
    if sps <= 0 or span <= 0:
        raise ValueError("span and sps must be positive")
    if beta < 0:
        raise ValueError("beta must be >= 0")

    n = span * sps
    t = (np.arange(-n / 2, n / 2 + 1, dtype=np.float64)) / sps  # length n+1

    if beta == 0.0:
        return np.sinc(t)

    denom = 1.0 - (2.0 * beta * t) ** 2
    h = np.empty_like(t)

    regular = np.abs(denom) > 1e-12
    h[regular] = np.sinc(t[regular]) * np.cos(np.pi * beta * t[regular]) / denom[regular]

    t0 = np.isclose(t, 0.0, atol=1e-12, rtol=0.0)
    h[t0] = 1.0

    tsing = np.isclose(np.abs(t), 1.0 / (2.0 * beta), atol=1e-12, rtol=0.0)
    if np.any(tsing):
        h[tsing] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))

    return h


def get_rcfilter(stride: int, n_rbs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (rc_filter, correction).
    rc_filter shape: (M,)
    correction shape: (M - ceil(M/2) - 1,)
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if n_rbs <= 0:
        raise ValueError("n_rbs must be positive")

    bw_factor = 10  # must be even
    roll_off = 0.2
    ff = _rcosdesign_normal(roll_off, n_rbs, bw_factor).astype(np.float64)
    l = ff.size

    kmax = int(np.floor((l / 2.0) / stride))
    offsets = np.arange(-kmax * stride, kmax * stride + 1, stride, dtype=np.int64)
    center_1based = (l + 1) // 2  # ceil(l/2)
    idx_0based = (offsets + center_1based) - 1
    rc_filter = ff[idx_0based]
    rc_filter = rc_filter / np.sum(rc_filter)

    tmp = np.cumsum(rc_filter)
    m = tmp.size
    start_1based = (m + 1) // 2
    start_0based = start_1based - 1
    correction = 1.0 / tmp[start_0based:-1]
    return rc_filter, correction


def compensate_cfo(
    rec_xpilots: np.ndarray,
    dmrs_symbols: np.ndarray,
    scs: float,
    cyclic_prefix_durations: np.ndarray,
    cfo_compensate: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If cfo_compensate == False, only estimates CFO.
    rec_xpilots shape: (n_re, n_dmrs_symbols, n_layers)
    CFO is estimated from the first two DMRS symbols; any additional DMRS symbols
    are phase-corrected (if enabled) using the estimated CFO.
    """
    rec_xpilots = np.asarray(rec_xpilots)
    dmrs_symbols = np.asarray(dmrs_symbols).astype(bool)
    cyclic_prefix_durations = np.asarray(cyclic_prefix_durations)

    n_dmrs_symbols = int(np.sum(dmrs_symbols))
    if n_dmrs_symbols < 2:
        return rec_xpilots, np.array([], dtype=np.float64)

    if rec_xpilots.ndim != 3:
        raise ValueError("rec_xpilots must have 3 dimensions (n_re, n_dmrs_symbols, n_layers)")

    n_layers = rec_xpilots.shape[2]
    cpds = cyclic_prefix_durations * scs

    dmrs_ix0 = np.flatnonzero(dmrs_symbols)
    dmrs1 = int(dmrs_ix0[0])
    dmrs2 = int(dmrs_ix0[1])
    n_syms = dmrs2 - dmrs1

    cfo_out = 0.0
    rec_for_cfo = rec_xpilots[:, :2, :]
    for i_layer in range(0, n_layers - 1, 2):
        cfo_cdm = np.vdot(rec_for_cfo[:, 0, i_layer], rec_for_cfo[:, 1, i_layer])
        cfo_cdm += np.vdot(rec_for_cfo[:, 0, i_layer + 1], rec_for_cfo[:, 1, i_layer + 1])
        cfo_out += np.angle(cfo_cdm)

    if (n_layers % 2) == 1:
        cfo_cdm = np.vdot(rec_for_cfo[:, 0, n_layers - 1], rec_for_cfo[:, 1, n_layers - 1])
        cfo_out += np.angle(cfo_cdm)

    n_samples = n_syms + float(np.sum(cpds[dmrs1 + 1 : dmrs2 + 1]))
    cfo_out = cfo_out / (2.0 * np.pi * n_samples) / int(np.ceil(n_layers / 2.0))

    if cfo_compensate:
        symbol_start_time = np.cumsum(np.concatenate(([cpds[0]], cpds[1:] + 1.0)))
        ph_correction = 2.0 * np.pi * symbol_start_time * cfo_out
        phase = np.exp(-1j * ph_correction[dmrs_ix0[:n_dmrs_symbols]])  # (n_dmrs_symbols,)
        rec_out = rec_xpilots * phase.reshape(1, n_dmrs_symbols, 1)
    else:
        rec_out = rec_xpilots

    return rec_out, np.array(cfo_out, dtype=np.float64)


def fill_ch_est_cdm(channel_in: np.ndarray, estimated: np.ndarray, hop: Hop, i_cdm: int) -> np.ndarray:
    """
    channel_in shape: (n_sc_total, n_sym_total, n_layers_total)
    estimated shape: (n_dmrs_re_total, n_layers_in_cdm)
    """
    NRE = 12
    channel_out = np.array(channel_in, copy=True)
    estimated = np.asarray(estimated)
    n_layers = estimated.shape[1]

    estimated_all = (np.nan + 1j * np.nan) * np.ones((hop.nPRBs * NRE, n_layers), dtype=np.complex128)
    mask_all = np.tile(np.asarray(hop.DMRSREmask)[:, i_cdm - 1].astype(bool), hop.nPRBs)
    estimated_all[mask_all, :] = estimated

    filled_indices = np.flatnonzero(mask_all)
    n_filled = filled_indices.size
    for i in range(n_filled - 1):
        fi = filled_indices[i]
        fj = filled_indices[i + 1]
        start0 = fi + 1
        stop0 = fj - 1
        if start0 > stop0:
            continue
        stride = stop0 - start0 + 1
        span = estimated_all[fj, :] - estimated_all[fi, :]
        t = (np.arange(1, stride + 1, dtype=np.float64) / (stride + 1.0))[:, None]
        estimated_all[start0 : stop0 + 1, :] = estimated_all[fi, :][None, :] + span[None, :] * t

    occupied_scs = np.arange(NRE * hop.PRBstart, NRE * (hop.PRBstart + hop.nPRBs), dtype=np.int64)
    occupied_syms = hop.startSymbol + np.arange(hop.nAllocatedSymbols, dtype=np.int64)

    for i_layer in range(n_layers):
        last_f = filled_indices[-1]
        first_f = filled_indices[0]
        estimated_all[last_f:, i_layer] = estimated_all[last_f, i_layer]
        estimated_all[: first_f + 1, i_layer] = estimated_all[first_f, i_layer]

        i_layer_true = i_layer + (i_cdm - 1) * 2  # 0-based
        channel_out[occupied_scs[:, None], occupied_syms[None, :], i_layer_true] = estimated_all[:, i_layer][:, None]

    return channel_out


def process_hop(
    hop: Hop,
    pilots_hop: np.ndarray,
    smoothing: str,
    *,
    received_rg: np.ndarray,
    beta_dmrs: float,
    scs: float,
    cyclic_prefix_durations: np.ndarray,
    cfo_compensate: bool,
    symbol_start_time: Optional[np.ndarray],
    channel_est_rg: np.ndarray,
    noise_est: float,
    rsrp: float,
    epre: float,
    time_alignment: float,
    cfo: Optional[float],
) -> Tuple[np.ndarray, float, float, float, float, Optional[float]]:
    """
    Port of nested processHop with explicit I/O.
    received_rg shape: (n_sc_total, n_sym_total)
    pilots_hop shape: (n_re, n_dmrs_symbols_in_hop, n_layers)
    """
    NRE = 12
    n_re, n_dmrs_symbols_hop, n_layers_hop = pilots_hop.shape
    n_cdm = int(np.ceil(n_layers_hop / 2.0))

    received_pilots = (np.nan + 1j * np.nan) * np.ones((n_re, n_dmrs_symbols_hop, n_cdm), dtype=np.complex128)
    rec_xpilots = (np.nan + 1j * np.nan) * np.ones_like(pilots_hop, dtype=np.complex128)

    dmrs_symbols_mask = np.asarray(hop.DMRSsymbols).astype(bool)
    n_dmrs_symbols_mask = int(np.sum(dmrs_symbols_mask))

    # CDM loop (also defines mask_res for later time-alignment, matching MATLAB behavior)
    mask_res = None
    mask_prbs = np.asarray(hop.maskPRBs).astype(bool).reshape(-1)

    for i_cdm in range(1, n_cdm + 1):
        dmrs_remask_col = np.asarray(hop.DMRSREmask)[:, i_cdm - 1].astype(bool).reshape(-1)
        mask_res = np.kron(mask_prbs.astype(np.int8), dmrs_remask_col.astype(np.int8)) > 0

        rp = received_rg[mask_res, :][:, dmrs_symbols_mask]
        if rp.shape != (n_re, n_dmrs_symbols_hop):
            raise ValueError("Shape mismatch extracting received pilots vs pilots_hop.")
        received_pilots[:, :, i_cdm - 1] = rp
        epre += float(np.sum(np.abs(rp) ** 2))

        first_layer = (i_cdm - 1) * 2
        last_layer = min(n_layers_hop, i_cdm * 2)
        rec_xpilots[:, :, first_layer:last_layer] = rp[:, :, None] * np.conj(pilots_hop[:, :, first_layer:last_layer])

    rec_xpilots_nocfo, cfo_hop_arr = compensate_cfo(
        rec_xpilots,
        dmrs_symbols_mask,
        scs / 1000.0,
        cyclic_prefix_durations,
        cfo_compensate,
    )
    cfo_hop = None if cfo_hop_arr.size == 0 else float(cfo_hop_arr.reshape(()))
    if cfo_hop is not None:
        cfo = (cfo + cfo_hop) / 2.0 if cfo is not None else cfo_hop

    estimated_channel_p = np.sum(rec_xpilots_nocfo, axis=1) / (beta_dmrs * float(n_dmrs_symbols_mask))
    # estimated_channel_p shape: (n_re, n_layers_hop)

    if n_layers_hop >= 2:
        odd = estimated_channel_p[0::2, :]
        even = estimated_channel_p[1::2, :]
        n_pairs = min(odd.shape[0], even.shape[0])
        avg = (odd[:n_pairs, :] + even[:n_pairs, :]) / 2.0
        estimated_channel_p[0 : 2 * n_pairs : 2, :] = avg
        estimated_channel_p[1 : 2 * n_pairs : 2, :] = avg

    dmrs_remask = np.asarray(hop.DMRSREmask).astype(bool)

    if smoothing == "mean":
        mean_per_layer = np.mean(estimated_channel_p, axis=0, keepdims=True)
        estimated_channel_p = np.ones_like(estimated_channel_p) * mean_per_layer
    elif smoothing == "filter":
        sum_dmrs_col1 = int(np.sum(dmrs_remask[:, 0]))
        stride = int(NRE / sum_dmrs_col1)
        rc_filter, _ = get_rcfilter(stride, min(3, int(np.sum(mask_prbs))))

        if int(np.sum(mask_prbs)) > 1:
            n_pils = min(12, int(np.floor(len(rc_filter) / 2)))
        else:
            n_pils = sum_dmrs_col1

        for i_layer in range(n_layers_hop):
            v_begin = create_virtual_pilots(estimated_channel_p[:n_pils, i_layer], n_pils)
            v_end = create_virtual_pilots(estimated_channel_p[::-1][:n_pils, i_layer], n_pils)
            sig = np.concatenate([v_begin, estimated_channel_p[:, i_layer], np.flipud(v_end)])
            tmp = np.convolve(sig, rc_filter, mode="same")
            estimated_channel_p[:, i_layer] = tmp[n_pils : tmp.size - n_pils]
    elif smoothing == "none":
        pass
    else:
        raise ValueError(f"Unknown smoothing strategy {smoothing!r}.")

    # Time alignment (uses last mask_res, matching MATLAB behavior)
    if mask_res is None:
        raise RuntimeError("mask_res not computed.")
    est_channel_sc = np.zeros((mask_res.size, n_layers_hop), dtype=np.complex128)
    est_channel_sc[mask_res, :] = estimated_channel_p

    fft_size = 4096
    channel_ir_lp = np.sum(np.abs(np.fft.ifft(est_channel_sc, n=fft_size, axis=0)) ** 2, axis=1)
    half_cp_len = int(np.floor((144 / 2) * fft_size / 2048))

    max_delay = float(np.max(channel_ir_lp[:half_cp_len]))
    i_max_delay = int(np.argmax(channel_ir_lp[:half_cp_len]))
    tail = channel_ir_lp[-half_cp_len:]
    max_advance = float(np.max(tail))
    i_max_advance = int(np.argmax(tail))

    if max_delay >= max_advance:
        i_max = i_max_delay
    else:
        i_max_advance_1based = i_max_advance + 1
        i_max = -(half_cp_len - i_max_advance_1based + 1)

    time_alignment += float(i_max) / float(fft_size) / float(scs)

    estimated_rx = np.zeros_like(received_pilots, dtype=np.complex128)

    for i_cdm in range(1, n_cdm + 1):
        min_layer = (i_cdm - 1) * 2
        max_layer = min(i_cdm * 2, n_layers_hop)

        if cfo_compensate and (cfo_hop is not None):
            if symbol_start_time is None:
                raise ValueError("symbol_start_time required when CFO compensation is enabled.")
            ph_correction = 2.0 * np.pi * symbol_start_time * cfo_hop
            ph = np.exp(1j * ph_correction[dmrs_symbols_mask]).reshape(1, -1)
            for i_layer in range(min_layer, max_layer):
                term = estimated_channel_p[:, i_layer].reshape(-1, 1) * ph
                estimated_rx[:, :, i_cdm - 1] += beta_dmrs * pilots_hop[:, :, i_layer] * term
        else:
            for i_layer in range(min_layer, max_layer):
                estimated_rx[:, :, i_cdm - 1] += beta_dmrs * pilots_hop[:, :, i_layer] * estimated_channel_p[:, i_layer].reshape(-1, 1)

        estimated_channel_p_cdm = estimated_channel_p[:, min_layer:max_layer]
        channel_est_rg = fill_ch_est_cdm(channel_est_rg, estimated_channel_p_cdm, hop, i_cdm)

    noise_est += float(np.sum(np.abs(received_pilots - estimated_rx) ** 2))
    rsrp += float((beta_dmrs ** 2) * np.sum(np.abs(estimated_channel_p) ** 2) * n_dmrs_symbols_mask)

    return channel_est_rg, noise_est, rsrp, epre, time_alignment, cfo


# -----------------------------
# Final: srs_channel_estimator (requested)
# -----------------------------
def srs_channel_estimator(
    received_rg: np.ndarray,
    pilots: np.ndarray,
    beta_dmrs: float,
    hop1: Hop,
    hop2: Hop,
    config: Config,
) -> Tuple[np.ndarray, float, float, float, float, np.ndarray]:
    """
    Python port of MATLAB srsChannelEstimator.

    Inputs:
      received_rg: complex, shape (n_sc_total, n_sym_total)
      pilots: complex, shape (n_re, n_dmrs_symbols_total, n_layers)
      beta_dmrs: float
      hop1, hop2: Hop
      config: Config with scs, CyclicPrefixDurations, optional Smoothing, CFOCompensate

    Outputs:
      channel_est_rg: complex, shape (n_sc_total, n_sym_total, n_layers)
      noise_est: float
      rsrp: float
      epre: float
      time_alignment: float
      cfo_hz: np.ndarray scalar (shape ()) or empty array (shape (0,))
    """
    received_rg = np.asarray(received_rg)
    pilots = np.asarray(pilots)

    # Pilots has as many slices (third dimension) as the number of layers.
    n_layers = pilots.shape[2]

    cfo_compensate = True
    if config.CFOCompensate is not None:
        cfo_compensate = bool(config.CFOCompensate)

    # channelEstRG = complex(zeros([size(receivedRG), nLayers]));
    channel_est_rg = np.zeros((received_rg.shape[0], received_rg.shape[1], n_layers), dtype=np.complex128)

    noise_est = 0.0
    rsrp = 0.0
    epre = 0.0
    time_alignment = 0.0
    cfo: Optional[float] = None  # normalized CFO (MATLAB keeps [] until estimated)

    n_pilot_symbols_hop1 = int(np.sum(np.asarray(hop1.DMRSsymbols).astype(bool)))
    scs = float(config.scs)

    smoothing = config.Smoothing if (config.Smoothing is not None) else "filter"

    symbol_start_time: Optional[np.ndarray] = None
    if cfo_compensate:
        # CyclicPrefixDurations = config.CyclicPrefixDurations * scs / 1000;
        # symbolStartTime = cumsum([CPDs(1) CPDs(2:14) + 1]);
        cyclic_prefix_durations = np.asarray(config.CyclicPrefixDurations, dtype=np.float64) * scs / 1000.0
        symbol_start_time = np.cumsum(np.concatenate(([cyclic_prefix_durations[0]], cyclic_prefix_durations[1:] + 1.0)))
    else:
        cyclic_prefix_durations = np.asarray(config.CyclicPrefixDurations, dtype=np.float64)  # still needed by CFO estimator call

    # processHop(hop1, pilots(:, 1:nPilotSymbolsHop1, :), smoothing);
    channel_est_rg, noise_est, rsrp, epre, time_alignment, cfo = process_hop(
        hop1,
        pilots[:, :n_pilot_symbols_hop1, :],
        smoothing,
        received_rg=received_rg,
        beta_dmrs=beta_dmrs,
        scs=scs,
        cyclic_prefix_durations=np.asarray(config.CyclicPrefixDurations, dtype=np.float64),
        cfo_compensate=cfo_compensate,
        symbol_start_time=symbol_start_time,
        channel_est_rg=channel_est_rg,
        noise_est=noise_est,
        rsrp=rsrp,
        epre=epre,
        time_alignment=time_alignment,
        cfo=cfo,
    )

    all_dmrs_symbols = np.asarray(hop1.DMRSsymbols).astype(bool).copy()

    # if ~isempty(hop2.DMRSsymbols)
    hop2_dmrs = np.asarray(hop2.DMRSsymbols)
    if hop2_dmrs.size != 0 and bool(np.any(hop2_dmrs)):
        hop2_dmrs_bool = hop2_dmrs.astype(bool)
        if np.any(all_dmrs_symbols & hop2_dmrs_bool):
            raise AssertionError("Hops should not overlap.")
        all_dmrs_symbols = all_dmrs_symbols | hop2_dmrs_bool

        if not np.all(np.asarray(hop1.DMRSREmask) == np.asarray(hop2.DMRSREmask)):
            raise AssertionError("The DM-RS mask should be the same for the two hops.")

        channel_est_rg, noise_est, rsrp, epre, time_alignment, cfo = process_hop(
            hop2,
            pilots[:, n_pilot_symbols_hop1:, :],
            smoothing,
            received_rg=received_rg,
            beta_dmrs=beta_dmrs,
            scs=scs,
            cyclic_prefix_durations=np.asarray(config.CyclicPrefixDurations, dtype=np.float64),
            cfo_compensate=cfo_compensate,
            symbol_start_time=symbol_start_time,
            channel_est_rg=channel_est_rg,
            noise_est=noise_est,
            rsrp=rsrp,
            epre=epre,
            time_alignment=time_alignment,
            cfo=cfo,
        )

    n_dmrs_symbols = int(np.sum(all_dmrs_symbols))
    n_pilots = hop1.nPRBs * int(np.sum(np.asarray(hop1.DMRSREmask)[:, 0].astype(bool))) * n_dmrs_symbols

    rsrp = rsrp / float(n_pilots) / float(n_layers)
    epre = epre / float(n_pilots)

    # noiseEst = noiseEst / (ceil(nLayers / 2) * nPilots - 1);
    noise_est = noise_est / (float(np.ceil(n_layers / 2.0)) * float(n_pilots) - 1.0)

    # if ~isempty(hop2.DMRSsymbols) timeAlignment = timeAlignment / 2;
    if hop2_dmrs.size != 0 and bool(np.any(hop2_dmrs)):
        time_alignment = time_alignment / 2.0

    # if (cfoCompensate && ~isempty(cfo)) apply PH correction to channel estimates
    if cfo_compensate and (cfo is not None):
        if symbol_start_time is None:
            raise ValueError("symbol_start_time missing while CFO compensation is enabled.")
        ph_correction = 2.0 * np.pi * symbol_start_time * cfo  # shape: (n_sym_total,)
        channel_est_rg = channel_est_rg * np.exp(1j * ph_correction).reshape(1, -1, 1)

    # Convert CFO from normalized units to hertz.
    if cfo is None:
        cfo_hz = np.array([], dtype=np.float64)
    else:
        cfo_hz = np.array(cfo * scs, dtype=np.float64)

    return channel_est_rg, float(noise_est), float(rsrp), float(epre), float(time_alignment), cfo_hz
