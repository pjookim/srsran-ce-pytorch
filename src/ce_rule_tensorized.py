from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ----------------------------
# Data containers (MATLAB struct 대응)
# ----------------------------
@dataclass
class HopConfig:
    DMRSsymbols: torch.Tensor        # (n_sym_total,) bool
    DMRSREmask: torch.Tensor         # (12, nCDM) bool
    PRBstart: int                    # 0-based
    nPRBs: int
    maskPRBs: torch.Tensor           # (n_prb_total,) bool
    startSymbol: int                 # 0-based
    nAllocatedSymbols: int


@dataclass
class EstimatorConfig:
    scs: float                       # Hz
    CyclicPrefixDurations: torch.Tensor  # (>=14,) ms
    Smoothing: str = "filter"
    CFOCompensate: bool = True


# ----------------------------
# Utilities you already have (paste your implementations)
# ----------------------------
def _unwrap_1d(ph: torch.Tensor) -> torch.Tensor:
    """
    MATLAB unwrap과 동일한 목적: 위상 점프(2pi 배수)를 제거하여 연속 위상으로 만듦.
    입력: (N,) real tensor
    출력: (N,) real tensor
    """
    if ph.numel() <= 1:
        return ph

    two_pi = 2.0 * math.pi
    pi = math.pi

    # dd = diff(ph)
    dd = ph[1:] - ph[:-1]

    # Wrap dd into [-pi, pi)
    ddmod = torch.remainder(dd + pi, two_pi) - pi

    # Special case: if ddmod == -pi and dd > 0, set ddmod = +pi (numpy.unwrap 관례)
    ddmod = torch.where((ddmod == -pi) & (dd > 0), ddmod + two_pi, ddmod)

    # correction that needs to be added cumulatively
    correction = ddmod - dd

    # If abs(dd) < pi, don't correct (already continuous enough)
    correction = torch.where(torch.abs(dd) < pi, torch.zeros_like(correction), correction)

    # cumulative correction, prepend 0 for first element
    cumcorr = torch.cumsum(correction, dim=0)
    cumcorr = torch.cat([torch.zeros(1, device=ph.device, dtype=ph.dtype), cumcorr], dim=0)

    return ph + cumcorr


def create_virtual_pilots(in_pilots: torch.Tensor, n_virtuals: int) -> torch.Tensor:
    """
    MATLAB:
      virtualPilots = createVirtualPilots(inPilots, nVirtuals)

    Args:
      in_pilots: (nPilots,) complex tensor (또는 complex로 변환 가능한 tensor)
      n_virtuals: 생성할 virtual pilot 개수 (>=0)

    Returns:
      virtual_pilots: (n_virtuals,) complex tensor
    """
    if n_virtuals < 0:
        raise ValueError("n_virtuals must be >= 0")
    if n_virtuals == 0:
        # MATLAB에서는 빈 컬럼 벡터가 나오지만, 여기서는 (0,) 반환
        return torch.empty((0,), dtype=torch.complex64, device=in_pilots.device)

    # Ensure tensor + complex dtype
    in_pilots = torch.as_tensor(in_pilots)
    if not torch.is_complex(in_pilots):
        in_pilots = in_pilots.to(torch.complex64)

    n_pilots = in_pilots.numel()
    if n_pilots == 0:
        raise ValueError("in_pilots must be non-empty")
    if n_pilots == 1:
        # 분모(normx - nPilots*mx^2)가 0이 되므로 MATLAB도 불안정.
        # 여기서는 MATLAB 수식 그대로는 불가능하므로, 안전하게 상수 외삽(동일 값)으로 처리.
        # (알고리즘 변경을 피하려면 입력이 최소 2개 이상이어야 함.)
        amp = torch.abs(in_pilots).repeat(n_virtuals)
        ph = torch.angle(in_pilots).repeat(n_virtuals)
        return amp * torch.exp(1j * ph)

    # x = 0:nPilots-1
    # MATLAB x는 double row vector. torch에서는 float로.
    x = torch.arange(n_pilots, device=in_pilots.device, dtype=torch.float64)
    mx = x.mean()
    normx = (x.norm() ** 2)  # = sum(x^2)

    # --- modulus fit ---
    y = torch.abs(in_pilots).to(torch.float64)
    my = y.mean()

    # a = (x*y - nPilots*mx*my) / (normx - nPilots*mx^2)
    # MATLAB x*y 는 dot product (1xn)*(nx1)
    xy = (x * y).sum()
    denom = (normx - n_pilots * (mx ** 2))
    a = (xy - n_pilots * mx * my) / denom
    b = my - a * mx

    # virtualPilots = a * (-nVirtuals:-1)' + b
    k = torch.arange(-n_virtuals, 0, device=in_pilots.device, dtype=torch.float64)  # [-nV, ..., -1]
    virtual_pilots = a * k + b  # amplitude (real)

    # --- phase fit ---
    y = torch.angle(in_pilots).to(torch.float64)
    y = _unwrap_1d(y)
    my = y.mean()

    xy = (x * y).sum()
    a = (xy - n_pilots * mx * my) / denom
    b = my - a * mx

    # virtualPilots = virtualPilots .* exp(1j * (a * (-nVirtuals:-1)' + b));
    ph = a * k + b
    # Match input complex precision (complex64/complex128)
    out_complex_dtype = in_pilots.dtype
    virtual_pilots = virtual_pilots.to(torch.float64) * torch.exp((1j * ph).to(torch.complex128))
    virtual_pilots = virtual_pilots.to(out_complex_dtype)

    return virtual_pilots


def _rcosdesign_normal(beta: float, span: int, sps: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """
    MATLAB rcosdesign(beta, span, sps, 'normal')에 해당하는 raised-cosine FIR 계수 생성.
    - span: filter span in symbols
    - sps : samples per symbol
    출력 길이: span*sps + 1
    주의: MATLAB 내부 정규화 방식과 완전히 동일한 '정규화'까지 보장하려고 하지 않음.
         (이 함수의 호출부에서 rcFilter를 sum=1로 다시 정규화하므로, 상대 형태가 핵심)
    """
    # Time base: t in symbol durations (T=1)
    # t = -span/2 : 1/sps : span/2  (length span*sps + 1)
    n = torch.arange(-span * sps // 2, span * sps // 2 + 1, device=device, dtype=dtype)
    t = n / float(sps)  # in symbol times

    pi = math.pi

    # sinc(t) = sin(pi t) / (pi t), with sinc(0)=1
    sinc_t = torch.where(t == 0, torch.ones_like(t), torch.sin(pi * t) / (pi * t))

    # raised cosine:
    # h(t) = sinc(t) * cos(pi*beta*t) / (1 - (2*beta*t)^2)
    denom = 1.0 - (2.0 * beta * t) ** 2
    h = sinc_t * torch.cos(pi * beta * t) / denom

    # Handle singularities at t = ± 1/(2*beta) using the analytic limit:
    # lim_{t->1/(2β)} h(t) = (πβ/2) * sin(1/(2β))
    if beta > 0:
        t0 = 1.0 / (2.0 * beta)
        # since t is on a discrete grid, check exact equality in float is risky;
        # use a small tolerance based on step size.
        tol = (1.0 / float(sps)) * 1e-6
        mask = torch.isfinite(h) == 0  # denom hits 0 -> inf/nan
        # also catch near hits robustly
        mask = mask | (torch.abs(torch.abs(t) - t0) < tol)
        if mask.any():
            lim_val = (pi * beta / 2.0) * math.sin(1.0 / (2.0 * beta))
            h = torch.where(mask, torch.full_like(h, lim_val), h)

    return h


def get_rc_filter(stride: int, n_rbs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MATLAB:
      [rcFilter, correction] = getRCfilter(stride, nRBs)

    Args:
      stride: downsampling stride when selecting taps from the designed filter (>=1)
      n_rbs : filter span in RBs (MATLAB code uses this as 'span' for rcosdesign)

    Returns:
      rc_filter : (M,) real tensor, sum(rc_filter)=1
      correction: (K,) real tensor, tail correction factors
    """
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if n_rbs <= 0:
        raise ValueError("n_rbs must be >= 1")

    bw_factor = 10  # must be even (as MATLAB comment)
    roll_off = 0.2

    # ff = rcosdesign(rollOff, nRBs, bwFactor, 'normal')';
    ff = _rcosdesign_normal(roll_off, n_rbs, bw_factor, dtype=torch.float64)

    l = ff.numel()

    # MATLAB:
    # n = (-floor(l/2/stride)*stride:stride:floor(l/2/stride)*stride) + ceil(l/2);
    # rcFilter = ff(n);
    # (MATLAB indices are 1-based; translate to 0-based)
    half = l // 2  # since l is odd, half == floor(l/2)
    kmax = (half // stride) * stride  # floor(l/2/stride)*stride
    ks = torch.arange(-kmax, kmax + 1, stride, device=ff.device, dtype=torch.int64)

    center0 = (l - 1) // 2  # 0-based center == ceil(l/2)-1 for odd l
    idx0 = ks + center0
    rc_filter = ff[idx0].clone()

    # rcFilter = rcFilter / sum(rcFilter);
    rc_filter = rc_filter / rc_filter.sum()

    # tmp = cumsum(rcFilter);
    tmp = torch.cumsum(rc_filter, dim=0)

    # correction = 1./tmp(ceil(length(tmp)/2):end-1);
    m = tmp.numel()
    mid0 = math.ceil(m / 2) - 1  # 0-based
    # up to end-1 (exclude last element): tmp[mid0 : m-1]
    correction = 1.0 / tmp[mid0 : m - 1]

    return rc_filter, correction


def fill_ch_est_cdm(
    channel_in: torch.Tensor,
    estimated: torch.Tensor,
    hop,
    i_cdm: int,
) -> torch.Tensor:
    """
    MATLAB:
      channelOut = fillChEstCDM(channelIn, estimated, hop, iCDM)

    Args:
      channel_in: (n_sc, n_sym, n_layers_total) complex tensor (resource grid)
      estimated : (n_dmrs_re, n_layers_in_cdm) complex tensor
                  - MATLAB에서 size(estimated,2)가 nLayers
      hop: object with fields:
           - nPRBs (int)
           - DMRSREmask (bool tensor, shape [12, n_cdm_groups] or compatible)
           - PRBstart (int, 0-based)
           - startSymbol (int, 0-based)
           - nAllocatedSymbols (int)
      i_cdm: CDM group index (MATLAB과 동일하게 1-based로 넣는 것을 가정)

    Returns:
      channel_out: channel_in과 동일 shape의 complex tensor
    """
    NRE = 12

    # 타입/디바이스 정리
    channel_out = channel_in.clone()
    if not torch.is_complex(channel_out):
        channel_out = channel_out.to(torch.complex64)

    estimated = torch.as_tensor(estimated, device=channel_out.device)
    # Force dtype match with channel tensor to avoid index_put type mismatch (e.g., complex64 vs complex128)
    if not torch.is_complex(estimated):
        estimated = estimated.to(channel_out.dtype)
    else:
        estimated = estimated.to(channel_out.dtype)

    n_layers = estimated.shape[1]

    # estimatedAll = complex(nan(hop.nPRBs * NRE, nLayers));
    n_sc_hop = int(hop.nPRBs) * NRE
    nan_real = torch.tensor(float("nan"), device=channel_out.device, dtype=torch.float64)
    nan_c = (nan_real + 1j * nan_real).to(channel_out.dtype)
    estimated_all = torch.full((n_sc_hop, n_layers), nan_c, device=channel_out.device, dtype=channel_out.dtype)

    # maskAll = repmat(hop.DMRSREmask(:, iCDM), hop.nPRBs, 1);
    # MATLAB iCDM는 1-based
    cdm_col = i_cdm - 1
    dmrs_re_mask_col = hop.DMRSREmask[:, cdm_col]  # shape (12,)
    mask_all = dmrs_re_mask_col.repeat(int(hop.nPRBs))  # shape (nPRBs*12,)
    mask_all = mask_all.to(dtype=torch.bool, device=channel_out.device)

    # estimatedAll(maskAll, :) = estimated;
    estimated_all[mask_all, :] = estimated

    # filledIndices = find(maskAll); (1-based)  -> torch에서는 0-based
    filled_idx = torch.nonzero(mask_all, as_tuple=False).flatten()
    n_filled = int(filled_idx.numel())
    if n_filled == 0:
        # 채울 DMRS RE가 없으면 그대로 반환
        return channel_out

    # occupiedSCs = (NRE * hop.PRBstart):(NRE * (hop.PRBstart + hop.nPRBs) - 1);
    sc_start = NRE * int(hop.PRBstart)
    sc_stop = NRE * (int(hop.PRBstart) + int(hop.nPRBs))  # python slice end는 exclusive
    occupied_scs = torch.arange(sc_start, sc_stop, device=channel_out.device, dtype=torch.long)

    # occupiedSymbols = hop.startSymbol + (0:hop.nAllocatedSymbols-1);
    sym_start = int(hop.startSymbol)
    sym_stop = sym_start + int(hop.nAllocatedSymbols)
    occupied_syms = torch.arange(sym_start, sym_stop, device=channel_out.device, dtype=torch.long)

    # Vectorized linear interpolation/extrapolation for all subcarriers.
    pos = torch.arange(n_sc_hop, device=channel_out.device, dtype=torch.long)
    first_idx = filled_idx[0]
    last_idx = filled_idx[-1]
    left_mask = pos <= first_idx
    right_mask = pos >= last_idx
    mid_mask = ~(left_mask | right_mask)

    interp_all = torch.empty_like(estimated_all)
    interp_all[left_mask, :] = estimated_all[first_idx : first_idx + 1, :]
    interp_all[right_mask, :] = estimated_all[last_idx : last_idx + 1, :]

    if bool(mid_mask.any()):
        # right anchor index for each interior RE.
        right_anchor_ord = torch.searchsorted(filled_idx, pos[mid_mask], right=False)
        left_anchor_ord = right_anchor_ord - 1
        left_anchor_pos = filled_idx[left_anchor_ord]
        right_anchor_pos = filled_idx[right_anchor_ord]

        left_val = estimated_all[left_anchor_pos, :]
        right_val = estimated_all[right_anchor_pos, :]

        real_dtype = channel_out.real.dtype
        alpha = (
            (pos[mid_mask].to(real_dtype) - left_anchor_pos.to(real_dtype))
            / (right_anchor_pos.to(real_dtype) - left_anchor_pos.to(real_dtype))
        ).unsqueeze(1)
        interp_all[mid_mask, :] = left_val + alpha * (right_val - left_val)

    estimated_all = interp_all

    # iLayerTrue = iLayer + (iCDM - 1) * 2;  (1-based)
    layer_true0 = torch.arange(n_layers, device=channel_out.device, dtype=torch.long) + (i_cdm - 1) * 2

    # Replicate along symbol dimension with one matrix multiply.
    ones_sym = torch.ones((1, occupied_syms.numel()), device=channel_out.device, dtype=channel_out.dtype)
    block = torch.matmul(estimated_all.unsqueeze(-1), ones_sym).permute(0, 2, 1)
    channel_out[
        occupied_scs[:, None, None],
        occupied_syms[None, :, None],
        layer_true0[None, None, :],
    ] = block

    return channel_out


def compensate_cfo(
    rec_x_pilots: torch.Tensor,
    dmrs_symbols: torch.Tensor,
    scs: float,
    cyclic_prefix_durations: torch.Tensor,
    cfo_compensate: bool,
):
    """
    MATLAB:
      [recXpilotsOut, cfoOut] = compensateCFO(recXpilots, DMRSsymbols, SCS, CyclicPrefixDurations, cfoCompensate)

    Args:
      rec_x_pilots: complex tensor, shape (n_re, 2, n_layers)
                   - MATLAB code uses recXpilots(:, 1, iLayer) and (:, 2, iLayer)
      dmrs_symbols: logical mask over OFDM symbols (1D bool/int tensor)
      scs: subcarrier spacing in Hz (scalar)
      cyclic_prefix_durations: CP durations in milliseconds (1D tensor, expected length >= 14 in MATLAB code)
      cfo_compensate: if False, only estimates CFO (no correction applied)

    Returns:
      rec_x_pilots_out: corrected (or original) rec_x_pilots
      cfo_out: scalar tensor (float) if estimated, else empty tensor (shape (0,))
    """
    rec_x_pilots = torch.as_tensor(rec_x_pilots)
    dmrs_symbols = torch.as_tensor(dmrs_symbols)
    cyclic_prefix_durations = torch.as_tensor(cyclic_prefix_durations, device=rec_x_pilots.device)

    if not torch.is_complex(rec_x_pilots):
        rec_x_pilots = rec_x_pilots.to(torch.complex64)

    # if sum(DMRSsymbols) < 2: return input, []
    if int(dmrs_symbols.to(torch.int64).sum().item()) < 2:
        rec_x_pilots_out = rec_x_pilots
        cfo_out = torch.empty((0,), device=rec_x_pilots.device, dtype=torch.float64)
        return rec_x_pilots_out, cfo_out

    n_layers = rec_x_pilots.shape[2]

    # CPDs = CyclicPrefixDurations * SCS;
    # (MATLAB: ms * Hz => samples per OFDM symbol excluding FFT samples? 그대로 사용)
    CPDs = cyclic_prefix_durations.to(torch.float64) * float(scs)

    # dmrsIx = find(DMRSsymbols);  (MATLAB 1-based indices)
    dmrs_ix0 = torch.nonzero(dmrs_symbols.to(torch.bool), as_tuple=False).flatten()  # 0-based
    # MATLAB: nSyms = dmrsIx(2) - dmrsIx(1);
    n_syms = int((dmrs_ix0[1] - dmrs_ix0[0]).item())

    # Compute all layer-wise inner products at once:
    # rec(:,1,layer)' * rec(:,2,layer) = sum(conj(x1) * x2).
    inner_per_layer = torch.sum(torch.conj(rec_x_pilots[:, 0, :]) * rec_x_pilots[:, 1, :], dim=0)

    # cfoOut accumulation over CDM pairs: for iLayer=1:2:nLayers-1 (MATLAB 1-based)
    # Two adjacent layers are combined into one CDM contribution.
    n_even = (n_layers // 2) * 2
    cfo_terms = []
    if n_even > 0:
        cfo_terms.append(torch.angle(inner_per_layer[:n_even].reshape(-1, 2).sum(dim=1)).to(torch.float64))
    if (n_layers % 2) == 1:
        cfo_terms.append(torch.angle(inner_per_layer[-1:]).to(torch.float64))
    cfo_acc = torch.cat(cfo_terms).sum() if cfo_terms else torch.zeros((), device=rec_x_pilots.device, dtype=torch.float64)

    # nSamples = nSyms + sum(CPDs((dmrsIx(1) + 1):dmrsIx(2)));
    # MATLAB dmrsIx are 1-based; slice (dmrsIx(1)+1):dmrsIx(2) inclusive
    # Convert using 0-based dmrs_ix0: start = dmrs_ix0[0]+1, stop_inclusive = dmrs_ix0[1]
    cp_sum = torch.sum(CPDs[int(dmrs_ix0[0].item()) + 1 : int(dmrs_ix0[1].item()) + 1])
    n_samples = float(n_syms) + float(cp_sum.item())

    # cfoOut = cfoOut / (2*pi*nSamples) / ceil(nLayers/2);
    denom_cdm = math.ceil(n_layers / 2)
    cfo_out = cfo_acc / (2.0 * math.pi * n_samples) / float(denom_cdm)

    if cfo_compensate:
        # symbolStartTime = cumsum([CPDs(1) CPDs(2:14) + 1]);
        # MATLAB indexing assumes at least 14 symbols worth CPDs.
        # Build vector length 14: [CPDs[0], CPDs[1:14]+1] then cumsum
        if CPDs.numel() < 14:
            raise ValueError("cyclic_prefix_durations must have length >= 14 to match MATLAB code.")
        vec = torch.empty((14,), device=rec_x_pilots.device, dtype=torch.float64)
        vec[0] = CPDs[0]
        vec[1:] = CPDs[1:14] + 1.0
        symbol_start_time = torch.cumsum(vec, dim=0)

        # PHcorrection = 2*pi*symbolStartTime*cfoOut;
        ph_correction = 2.0 * math.pi * symbol_start_time * cfo_out  # (14,)

        # recXpilotsOut = recXpilots .* reshape(exp(-1j * PHcorrection(dmrsIx)), 1, []);
        # MATLAB dmrsIx are 1-based; our dmrs_ix0 are 0-based, so direct gather is OK.
        ph_dmrs = ph_correction[dmrs_ix0.to(torch.long)]  # (n_dmrs_syms,)
        rot = torch.exp((-1j * ph_dmrs).to(torch.complex128)).to(rec_x_pilots.dtype)  # (n_dmrs_syms,)
        # Broadcast over (n_re, n_dmrs_syms, n_layers)
        rec_x_pilots_out = rec_x_pilots * rot.view(1, -1, 1)
    else:
        rec_x_pilots_out = rec_x_pilots

    return rec_x_pilots_out, cfo_out


def _fro_norm_sq(x: torch.Tensor) -> torch.Tensor:
    # ||x||_F^2
    return torch.sum(torch.abs(x) ** 2)


def _conv_same_1d_complex(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    MATLAB conv(x, h, "same") with x complex 1D, h real 1D.
    torch.nn.functional.conv1d is cross-correlation, so we flip h to match convolution.
    """
    x = torch.as_tensor(x)
    h = torch.as_tensor(h, device=x.device)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x and h must be 1D")
    if h.numel() == 0:
        return x.clone()

    # Ensure dtypes
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    if torch.is_complex(h):
        # MATLAB rcFilter is real; keep real part if complex accidentally
        h = h.real
    h = h.to(dtype=torch.float64)

    k = int(h.numel())
    pad = k // 2  # "same" assuming odd-ish; for even kernels MATLAB differs subtly but RC here is typically odd
    # Flip for convolution
    h_flip = torch.flip(h, dims=[0]).to(dtype=torch.float64)

    # conv1d expects (N, C, L)
    xr = x.real.to(torch.float64).view(1, 1, -1)
    xi = x.imag.to(torch.float64).view(1, 1, -1)
    w = h_flip.view(1, 1, -1)

    yr = F.conv1d(xr, w, padding=pad)
    yi = F.conv1d(xi, w, padding=pad)

    y = yr.view(-1) + 1j * yi.view(-1)
    return y.to(x.dtype)

def process_hop(
    hop_,
    pilots_: torch.Tensor,
    smoothing_: str,
    *,
    received_rg: torch.Tensor,
    scs: float,
    cyclic_prefix_durations: torch.Tensor,
    cfo_compensate: bool,
    beta_dmrs: float | torch.Tensor,
    symbol_start_time: torch.Tensor,
    # running accumulators / states (MATLAB outer-scope variables)
    epre: torch.Tensor,
    cfo,  # None or scalar tensor
    time_alignment: torch.Tensor,
    channel_est_rg: torch.Tensor,
    noise_est: torch.Tensor,
    rsrp: torch.Tensor,
    # required utility functions (ported earlier)
    compensate_cfo_fn,
    get_rc_filter_fn,
    create_virtual_pilots_fn,
    fill_ch_est_cdm_fn,
):
    """
    1:1 port of nested MATLAB function processHop(hop_, pilots_, smoothing_).

    Expected shapes (consistent with MATLAB usage):
      pilots_    : (n_dmrs_re, n_dmrs_syms, n_layers) complex
      received_rg: (n_sc_total, n_sym_total) complex
      channel_est_rg: (n_sc_total, n_sym_total, n_layers_total) complex

    hop_ fields used:
      hop_.maskPRBs        : (n_prbs_total,) bool
      hop_.DMRSREmask      : (12, n_cdm) bool
      hop_.DMRSsymbols     : (n_sym_total,) bool
      hop_.nPRBs           : int
      hop_.PRBstart        : int (0-based)
      hop_.startSymbol     : int (0-based)
      hop_.nAllocatedSymbols : int
    """
    NRE = 12

    pilots_ = torch.as_tensor(pilots_, device=received_rg.device)
    if not torch.is_complex(pilots_):
        pilots_ = pilots_.to(torch.complex64)

    received_rg = torch.as_tensor(received_rg, device=pilots_.device)
    if not torch.is_complex(received_rg):
        received_rg = received_rg.to(torch.complex64)

    beta_dmrs_t = torch.as_tensor(beta_dmrs, device=pilots_.device, dtype=torch.float64)
    symbol_start_time = torch.as_tensor(symbol_start_time, device=pilots_.device, dtype=torch.float64)
    cyclic_prefix_durations = torch.as_tensor(cyclic_prefix_durations, device=pilots_.device, dtype=torch.float64)

    # Number of layers and number of CDM groups (2 layers per group).
    n_layers_ = pilots_.shape[2]
    n_cdm_ = int(math.ceil(n_layers_ / 2))

    # receivedPilots_ = complex(nan([size(pilots_, [1 2]), nCDM_]));
    nan_real = torch.tensor(float("nan"), device=pilots_.device, dtype=torch.float64)
    nan_c = (nan_real + 1j * nan_real).to(pilots_.dtype)
    received_pilots_ = torch.full((pilots_.shape[0], pilots_.shape[1], n_cdm_), nan_c,
                                 device=pilots_.device, dtype=pilots_.dtype)

    # recXpilots_ = complex(nan(size(pilots_)));
    rec_x_pilots_ = torch.full_like(pilots_, nan_c)

    mask_prbs_ = torch.as_tensor(hop_.maskPRBs, device=pilots_.device, dtype=torch.bool)
    dmrs_symbols_mask = torch.as_tensor(hop_.DMRSsymbols, device=pilots_.device, dtype=torch.bool)

    # Precompute dmrs symbol indices (for slicing)
    dmrs_sym_idx0 = torch.nonzero(dmrs_symbols_mask, as_tuple=False).flatten()
    n_dmrs_symbols_ = int(dmrs_symbols_mask.to(torch.int64).sum().item())

    # Main loop over CDM groups
    for i_cdm_ in range(1, n_cdm_ + 1):
        # maskREs_ = (kron(maskPRBs_, hop_.DMRSREmask(:, iCDM_)) > 0);
        dmrs_re_mask_col = torch.as_tensor(hop_.DMRSREmask[:, i_cdm_ - 1], device=pilots_.device, dtype=torch.bool)
        # kron for boolean: repeat each PRB mask element by 12 and AND with re mask pattern
        # Equivalent to kron(maskPRBs_, dmrs_re_mask_col)
        mask_res_ = dmrs_re_mask_col.repeat(mask_prbs_.numel()) & mask_prbs_.repeat_interleave(NRE)

        # Pick the REs corresponding to the pilots.
        # receivedRG(maskREs_, hop_.DMRSsymbols)
        rx_sel = received_rg[mask_res_, :][:, dmrs_sym_idx0]  # (n_dmrs_re, n_dmrs_symbols)
        received_pilots_[:, :, i_cdm_ - 1] = rx_sel

        # Compute receive side DM-RS EPRE.
        epre = epre + _fro_norm_sq(rx_sel)

        # LSE-estimate recXpilots_ for layers in this CDM.
        first_layer_ = (i_cdm_ - 1) * 2 + 1  # 1-based
        last_layer_ = min(n_layers_, i_cdm_ * 2)  # 1-based
        fl0 = first_layer_ - 1
        ll0 = last_layer_ - 1

        # recXpilots_(..., first:last) = receivedPilots * conj(pilots_(..., first:last))
        rec_x_pilots_[:, :, fl0:ll0 + 1] = rx_sel.unsqueeze(2) * torch.conj(pilots_[:, :, fl0:ll0 + 1])

    # CFO estimate/compensation (if cfoCompensate==false, only estimates CFO)
    rec_x_pilots_nocfo_, cfo_hop_ = compensate_cfo_fn(
        rec_x_pilots_,
        dmrs_symbols_mask,
        scs / 1000.0,
        cyclic_prefix_durations,
        cfo_compensate,
    )

    # MATLAB: if ~isempty(cfoHop_) ...
    if isinstance(cfo_hop_, torch.Tensor) and cfo_hop_.numel() != 0:
        if cfo is not None and isinstance(cfo, torch.Tensor) and cfo.numel() != 0:
            cfo = (cfo + cfo_hop_) / 2
        else:
            cfo = cfo_hop_

    # estimatedChannelP_ = squeeze(sum(recXpilotsNOCFO_, 2) / betaDMRS / nDMRSsymbols_);
    # sum over DMRS symbols dimension (dim=1 in Python, corresponds to MATLAB dim 2)
    estimated_channel_p_ = torch.sum(rec_x_pilots_nocfo_, dim=1) / beta_dmrs_t / float(n_dmrs_symbols_)
    # squeeze: keep as (n_dmrs_re, n_layers) already
    # (if n_layers==1, this is still 2D; MATLAB squeeze would make it 1D, but later code expects 2D in practice)
    if estimated_channel_p_.ndim == 1:
        estimated_channel_p_ = estimated_channel_p_.unsqueeze(1)

    # CDM interference removal for >=2 layers
    if n_layers_ >= 2:
        # estimatedChannelP_(1:2:end,:) = (odd + even)/2; even = odd;
        odd = estimated_channel_p_[0::2, :]
        even = estimated_channel_p_[1::2, :]
        m = min(odd.shape[0], even.shape[0])
        if m > 0:
            avg = (odd[:m, :] + even[:m, :]) / 2
            estimated_channel_p_[0:2 * m:2, :] = avg
            estimated_channel_p_[1:2 * m:2, :] = avg

    DMRSREmask_ = torch.as_tensor(hop_.DMRSREmask, device=pilots_.device, dtype=torch.bool)

    # Smoothing switch
    if smoothing_ == "mean":
        # estimatedChannelP_ = ones(size(...)) .* mean(...,1)
        mean_col = torch.mean(estimated_channel_p_, dim=0, keepdim=True)  # (1, n_layers)
        estimated_channel_p_ = torch.ones_like(estimated_channel_p_) * mean_col
    elif smoothing_ == "filter":
        # rcFilter_ = getRCfilter(12/sum(DMRSREmask_(:, 1)), min(3, sum(maskPRBs_)));
        dmrs_per_prb = int(DMRSREmask_[:, 0].to(torch.int64).sum().item())
        stride_ = int(12 // dmrs_per_prb)  # MATLAB assumes nice divisibility here
        n_rbs_span = int(min(3, int(mask_prbs_.to(torch.int64).sum().item())))
        rc_filter_, _ = get_rc_filter_fn(stride_, n_rbs_span)

        if int(mask_prbs_.to(torch.int64).sum().item()) > 1:
            n_pils_ = int(min(12, int(math.floor(rc_filter_.numel() / 2))))
        else:
            n_pils_ = int(dmrs_per_prb)

        for i_layer_ in range(1, n_layers_ + 1):
            il0 = i_layer_ - 1

            # vPilsBegin_ = createVirtualPilots(estimatedChannelP_(1:nPils_, iLayer_), nPils_);
            v_begin = create_virtual_pilots_fn(estimated_channel_p_[:n_pils_, il0], n_pils_)

            # vPilsEnd_ = createVirtualPilots(estimatedChannelP_(end:-1:end-nPils_+1, iLayer_), nPils_);
            tail_rev = torch.flip(estimated_channel_p_[-n_pils_:, il0], dims=[0])
            v_end = create_virtual_pilots_fn(tail_rev, n_pils_)

            # tmp_ = conv([vPilsBegin_; estimatedChannelP_(:, iLayer_); flipud(vPilsEnd_)], rcFilter_, "same");
            x = torch.cat([v_begin, estimated_channel_p_[:, il0], torch.flip(v_end, dims=[0])], dim=0)
            tmp = _conv_same_1d_complex(x, rc_filter_)

            # estimatedChannelP_(:, iLayer_) = tmp_(nPils_+1:end-nPils_);
            estimated_channel_p_[:, il0] = tmp[n_pils_ : tmp.numel() - n_pils_]
    elif smoothing_ == "none":
        pass
    else:
        raise ValueError(f"Unknown smoothing strategy {smoothing_}.")

    # --- Estimate time alignment ---
    # estChannelSC_ = complex(zeros(length(maskREs_), nLayers_));
    # NOTE: maskREs_ here refers to the LAST computed mask_res_ (MATLAB scope behavior).
    # In practice for DMRSREmask patterns, it is the same length each time; we mirror MATLAB behavior.
    est_channel_sc_ = torch.zeros((mask_res_.numel(), n_layers_), device=pilots_.device, dtype=pilots_.dtype)
    est_channel_sc_[mask_res_, :] = estimated_channel_p_

    fft_size_ = 4096
    # channelIRlp_ = sum(abs(ifft(estChannelSC_, fftSize_)).^2, 2);
    ir = torch.fft.ifft(est_channel_sc_, n=fft_size_, dim=0)
    channel_ir_lp_ = torch.sum(torch.abs(ir) ** 2, dim=1)  # (fft_size_,) real

    half_cp_len_ = int(math.floor((144 / 2) * fft_size_ / 2048))
    head = channel_ir_lp_[:half_cp_len_]
    tail = channel_ir_lp_[-half_cp_len_:]

    max_delay_, i_max_delay0 = torch.max(head, dim=0)
    max_adv_, i_max_adv0 = torch.max(tail, dim=0)

    # MATLAB indices are 1-based:
    i_max_delay = int(i_max_delay0.item()) + 1
    i_max_adv = int(i_max_adv0.item()) + 1

    if float(max_delay_.item()) >= float(max_adv_.item()):
        i_max_ = i_max_delay - 1
    else:
        i_max_ = -(half_cp_len_ - i_max_adv + 1)

    time_alignment = time_alignment + (float(i_max_) / float(fft_size_) / float(scs))

    # --- Reconstruct estimated RX on pilots, fill channel grid, noise/rsrp ---
    estimated_rx_ = torch.zeros_like(received_pilots_)  # (n_dmrs_re, n_dmrs_symbols, n_cdm)
    beta_scale = beta_dmrs_t.to(pilots_.real.dtype)

    for i_cdm_ in range(1, n_cdm_ + 1):
        min_layer_ = (i_cdm_ - 1) * 2 + 1
        max_layer_ = min(i_cdm_ * 2, n_layers_)
        min0 = min_layer_ - 1
        max0 = max_layer_ - 1
        layer_slice = slice(min0, max0 + 1)
        pilots_cdm = pilots_[:, :, layer_slice]  # (n_re, n_dmrs_syms, n_layers_cdm)
        h_cdm = estimated_channel_p_[:, layer_slice]  # (n_re, n_layers_cdm)

        if cfo_compensate and isinstance(cfo_hop_, torch.Tensor) and cfo_hop_.numel() != 0:
            # PHcorrection = 2*pi*symbolStartTime*cfoHop_;
            ph_corr = 2.0 * math.pi * symbol_start_time * cfo_hop_
            # exp(1j*PHcorrection(hop_.DMRSsymbols))
            ph_dmrs = torch.exp((1j * ph_corr[dmrs_symbols_mask]).to(torch.complex128)).to(pilots_.dtype)  # (n_dmrs_syms,)
            hsym = h_cdm.unsqueeze(1) * ph_dmrs.view(1, -1, 1)  # (n_re, n_dmrs_syms, n_layers_cdm)
        else:
            hsym = h_cdm.unsqueeze(1).expand(-1, n_dmrs_symbols_, -1)  # (n_re, n_dmrs_syms, n_layers_cdm)

        # Sum layer contributions for this CDM in one tensorized expression.
        estimated_rx_[:, :, i_cdm_ - 1] = beta_scale * torch.sum(pilots_cdm * hsym, dim=2)

        # Fill channel estimate on resource grid (linearly interpolate other subcarriers inside PRBs)
        estimated_channel_p_cdm_ = estimated_channel_p_[:, layer_slice]
        channel_est_rg = fill_ch_est_cdm_fn(channel_est_rg, estimated_channel_p_cdm_, hop_, i_cdm_)

    noise_est = noise_est + _fro_norm_sq(received_pilots_ - estimated_rx_)
    rsrp = rsrp + (beta_dmrs_t ** 2) * _fro_norm_sq(estimated_channel_p_) * float(n_dmrs_symbols_)

    return {
        "epre": epre,
        "cfo": cfo,
        "time_alignment": time_alignment,
        "channel_est_rg": channel_est_rg,
        "noise_est": noise_est,
        "rsrp": rsrp,
    }


# ----------------------------
# Main function: srsChannelEstimator
# ----------------------------
def srs_channel_estimator(
    received_rg: torch.Tensor,
    pilots: torch.Tensor,
    beta_dmrs: float | torch.Tensor,
    hop1: HopConfig,
    hop2: HopConfig,
    config: EstimatorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MATLAB:
      [channelEstRG, noiseEst, rsrp, epre, timeAlignment, cfo] =
         srsChannelEstimator(receivedRG, pilots, betaDMRS, hop1, hop2, config)

    Args:
      received_rg: (n_sc_total, n_sym_total) complex
      pilots:      (n_dmrs_re, n_dmrs_syms_total, n_layers) complex
      beta_dmrs:   scalar (linear amplitude gain) or tensor scalar
      hop1, hop2:  hop configs
      config:      estimator config

    Returns:
      channel_est_rg: (n_sc_total, n_sym_total, n_layers) complex
      noise_est:      scalar float tensor
      rsrp:           scalar float tensor
      epre:           scalar float tensor
      time_alignment: scalar float tensor
      cfo:            scalar float tensor (Hz) OR empty tensor if not estimated
    """
    received_rg = torch.as_tensor(received_rg)
    pilots = torch.as_tensor(pilots, device=received_rg.device)

    if not torch.is_complex(received_rg):
        received_rg = received_rg.to(torch.complex64)
    if not torch.is_complex(pilots):
        pilots = pilots.to(torch.complex64)

    # Pilots has as many slices (third dimension) as the number of layers.
    n_layers = pilots.shape[2]

    # cfoCompensate default true, overridable
    cfo_compensate = True
    if hasattr(config, "CFOCompensate"):
        cfo_compensate = bool(config.CFOCompensate)

    # Initialize outputs / accumulators
    channel_est_rg = torch.zeros((received_rg.shape[0], received_rg.shape[1], n_layers),
                                 device=received_rg.device, dtype=received_rg.dtype)
    noise_est = torch.zeros((), device=received_rg.device, dtype=torch.float64)
    rsrp = torch.zeros((), device=received_rg.device, dtype=torch.float64)
    epre = torch.zeros((), device=received_rg.device, dtype=torch.float64)
    time_alignment = torch.zeros((), device=received_rg.device, dtype=torch.float64)
    cfo: Optional[torch.Tensor] = None  # MATLAB cfo = []

    n_pilot_symbols_hop1 = int(torch.as_tensor(hop1.DMRSsymbols, device=received_rg.device, dtype=torch.int64).sum().item())

    scs = float(config.scs)

    # smoothing strategy
    if hasattr(config, "Smoothing") and config.Smoothing is not None:
        smoothing = str(config.Smoothing)
    else:
        smoothing = "filter"

    # symbolStartTime (needed in processHop for CFO compensation path)
    symbol_start_time = torch.empty((0,), device=received_rg.device, dtype=torch.float64)
    if cfo_compensate:
        # MATLAB:
        # CyclicPrefixDurations = config.CyclicPrefixDurations * scs / 1000;
        # symbolStartTime = cumsum([CPDs(1) CPDs(2:14) + 1]);
        CPDs = torch.as_tensor(config.CyclicPrefixDurations, device=received_rg.device, dtype=torch.float64) * scs / 1000.0
        if CPDs.numel() < 14:
            raise ValueError("config.CyclicPrefixDurations must have length >= 14 to match MATLAB code.")
        vec = torch.empty((14,), device=received_rg.device, dtype=torch.float64)
        vec[0] = CPDs[0]
        vec[1:] = CPDs[1:14] + 1.0
        symbol_start_time = torch.cumsum(vec, dim=0)

    # --- process hop1 ---
    pilots_hop1 = pilots[:, :n_pilot_symbols_hop1, :]
    st1 = process_hop(
        hop1, pilots_hop1, smoothing,
        received_rg=received_rg,
        scs=scs,
        cyclic_prefix_durations=torch.as_tensor(config.CyclicPrefixDurations, device=received_rg.device, dtype=torch.float64),
        cfo_compensate=cfo_compensate,
        beta_dmrs=beta_dmrs,
        symbol_start_time=symbol_start_time,
        epre=epre,
        cfo=cfo,
        time_alignment=time_alignment,
        channel_est_rg=channel_est_rg,
        noise_est=noise_est,
        rsrp=rsrp,
        compensate_cfo_fn=compensate_cfo,
        get_rc_filter_fn=get_rc_filter,
        create_virtual_pilots_fn=create_virtual_pilots,
        fill_ch_est_cdm_fn=fill_ch_est_cdm,
    )
    epre = st1["epre"]
    cfo = st1["cfo"]
    time_alignment = st1["time_alignment"]
    channel_est_rg = st1["channel_est_rg"]
    noise_est = st1["noise_est"]
    rsrp = st1["rsrp"]

    # allDMRSsymbols = hop1.DMRSsymbols;
    all_dmrs_symbols = torch.as_tensor(hop1.DMRSsymbols, device=received_rg.device, dtype=torch.bool)

    # hop2 processing (if not empty)
    hop2_dmrs = torch.as_tensor(hop2.DMRSsymbols, device=received_rg.device)
    has_hop2 = (hop2_dmrs.numel() != 0) and (int(hop2_dmrs.to(torch.int64).sum().item()) != 0)

    if has_hop2:
        hop2_dmrs_bool = hop2_dmrs.to(torch.bool)

        # assert(~any(hop1.DMRSsymbols & hop2.DMRSsymbols))
        overlap = torch.any(all_dmrs_symbols & hop2_dmrs_bool).item()
        assert not overlap, "Hops should not overlap."

        all_dmrs_symbols = all_dmrs_symbols | hop2_dmrs_bool

        # assert(all(hop1.DMRSREmask == hop2.DMRSREmask, 'all'))
        same_mask = torch.all(torch.as_tensor(hop1.DMRSREmask, device=received_rg.device) ==
                              torch.as_tensor(hop2.DMRSREmask, device=received_rg.device)).item()
        assert same_mask, "The DM-RS mask should be the same for the two hops."

        pilots_hop2 = pilots[:, n_pilot_symbols_hop1:, :]
        st2 = process_hop(
            hop2, pilots_hop2, smoothing,
            received_rg=received_rg,
            scs=scs,
            cyclic_prefix_durations=torch.as_tensor(config.CyclicPrefixDurations, device=received_rg.device, dtype=torch.float64),
            cfo_compensate=cfo_compensate,
            beta_dmrs=beta_dmrs,
            symbol_start_time=symbol_start_time,
            epre=epre,
            cfo=cfo,
            time_alignment=time_alignment,
            channel_est_rg=channel_est_rg,
            noise_est=noise_est,
            rsrp=rsrp,
            compensate_cfo_fn=compensate_cfo,
            get_rc_filter_fn=get_rc_filter,
            create_virtual_pilots_fn=create_virtual_pilots,
            fill_ch_est_cdm_fn=fill_ch_est_cdm,
        )
        epre = st2["epre"]
        cfo = st2["cfo"]
        time_alignment = st2["time_alignment"]
        channel_est_rg = st2["channel_est_rg"]
        noise_est = st2["noise_est"]
        rsrp = st2["rsrp"]

    # nDMRSsymbols = sum(allDMRSsymbols);
    n_dmrs_symbols = int(all_dmrs_symbols.to(torch.int64).sum().item())

    # nPilots = hop1.nPRBs * sum(hop1.DMRSREmask(:, 1)) * nDMRSsymbols;
    dmrs_per_prb = int(torch.as_tensor(hop1.DMRSREmask, device=received_rg.device)[:, 0].to(torch.int64).sum().item())
    n_pilots = int(hop1.nPRBs) * dmrs_per_prb * n_dmrs_symbols

    beta_dmrs_t = torch.as_tensor(beta_dmrs, device=received_rg.device, dtype=torch.float64)

    # rsrp = rsrp / nPilots / nLayers;
    rsrp = rsrp / float(n_pilots) / float(n_layers)

    # epre = epre / nPilots;
    epre = epre / float(n_pilots)

    # noiseEst = noiseEst / (ceil(nLayers / 2) * nPilots - 1);
    noise_den = (math.ceil(n_layers / 2) * n_pilots - 1)
    noise_est = noise_est / float(noise_den)

    # if hop2 exists: timeAlignment = timeAlignment / 2;
    if has_hop2:
        time_alignment = time_alignment / 2.0

    # CFO phase correction to channelEstRG
    if cfo_compensate and (cfo is not None) and isinstance(cfo, torch.Tensor) and (cfo.numel() != 0):
        ph_correction = 2.0 * math.pi * symbol_start_time * cfo  # (14,)
        rot = torch.exp((1j * ph_correction).to(torch.complex128)).to(channel_est_rg.dtype)  # (14,)

        # MATLAB: channelEstRG = channelEstRG .* reshape(exp(1j * PHcorrection), 1, []);
        # -> broadcast over (n_sc, n_sym, n_layers); apply over symbol dimension.
        # NOTE: MATLAB uses symbolStartTime length 14, so assumes slot has 14 symbols.
        channel_est_rg = channel_est_rg * rot.view(1, -1, 1)

    # Convert CFO from normalized units to hertz.
    if cfo is None or (isinstance(cfo, torch.Tensor) and cfo.numel() == 0):
        cfo_hz = torch.empty((0,), device=received_rg.device, dtype=torch.float64)
    else:
        cfo_hz = cfo * float(scs)

    return channel_est_rg, noise_est, rsrp, epre, time_alignment, cfo_hz
