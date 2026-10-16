"""
IPIX reader: dual-polarisation complex slow-time series for the cells the
blind-separation study uses.

WHY IPIX MATTERS HERE, AND NETRAD DID NOT
CCEBM Eq. (6) and CERBM both want N complex mixtures at the SAME pulses.
NetRAD records HH, VH and VV as separate acquisitions hours apart, so no
same-pulse dual-pol pair exists there and none was fabricated. IPIX
transmits alternating H/V within each sweep and records like and cross
returns for both, so HH and VV ARE available at the same sweep index.
This is the model the algorithms were designed for.

FILE LAYOUT (McMaster OHGR netCDF)
  adc_data  (nsweep, ntxpol=2, nrange, nadc=4)  int8
  nadc index order is given by the scalars adc_like_I / adc_like_Q /
  adc_cross_I / adc_cross_Q, which are INDICES into nadc, not offsets.
  txpol 0 transmits H  -> like = HH, cross = HV
  txpol 1 transmits V  -> like = VV, cross = VH

Only the cells the study needs are cached, as complex64: the cell under
test, one adjacent cell and eight reference cells. About 10 MB per file
against 15-27 MB for the raw file, and the raw set is never copied.
"""
from __future__ import annotations

import os

import numpy as np

PRF_HZ = 1000.0
REF_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)   # 14 range cells only
ADJ_CELL = 2
NEEDED = tuple(sorted({0, ADJ_CELL} | set(REF_OFFSETS)))


def meta(path):
    import scipy.io as sio
    f = sio.netcdf_file(path, "r", mmap=False)
    a = f._attributes
    d = f.dimensions

    def s(k):
        return (a.get(k, b"").decode(errors="replace")
                if isinstance(a.get(k), bytes) else a.get(k))
    return {
        "file": os.path.basename(path),
        "date": s("Data_collection_date"),
        "site": s("Site"), "campaign": s("Campaign"),
        "tx_polarisation": s("TX_polarization"),
        "n_sweep": int(d["nsweep"]), "n_range": int(d["nrange"]),
        "prf_hz": float(f.variables["PRF"].data),
        "rf_ghz": float(f.variables["RF_frequency"].data),
        "pulse_length_ns": float(f.variables["Pulse_length"].data),
        "unambig_velocity_ms": float(f.variables["Unambig_velocity"].data),
        "range_m": [float(v) for v in np.array(f.variables["range"].data)],
    }


def cell_slab(path, cachedir):
    """Return (slab, meta) where slab is (n_sweep, len(NEEDED), 2) complex64
    with the last axis (HH, VV). Cached atomically so a power cut cannot
    leave a half-written cache behind."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(cachedir, f"ipix_{stem}.npz")
    if os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=True)
            return z["slab"], z["meta"].item()
        except Exception:
            os.remove(cache)          # truncated by a crash; rebuild
    import scipy.io as sio
    f = sio.netcdf_file(path, "r", mmap=False)
    m = meta(path)
    n_range = m["n_range"]
    cut = n_range // 2
    cols = [cut + o for o in NEEDED]
    if min(cols) < 0 or max(cols) >= n_range:
        raise ValueError(f"reference offsets fall outside {n_range} cells")
    # The 1993 OHGR files carry adc_like_I / adc_like_Q as scalars giving
    # the index of each component inside nadc. The 1998 ANTSTEP files omit
    # them and use the conventional order, so fall back to that rather
    # than dropping two acquisitions.
    default = {"adc_like_I": 0, "adc_like_Q": 1}
    iq = {k: (int(f.variables[k].data) if k in f.variables else default[k])
          for k in ("adc_like_I", "adc_like_Q")}
    a = f.variables["adc_data"].data          # (sweep, txpol, range, adc)
    sub = a[:, :, cols, :].astype(np.float32)
    del a
    like = sub[..., iq["adc_like_I"]] + 1j * sub[..., iq["adc_like_Q"]]
    # txpol 0 -> HH, txpol 1 -> VV
    slab = np.stack([like[:, 0, :], like[:, 1, :]], axis=-1).astype(
        np.complex64)
    slab = slab - slab.mean(axis=0, keepdims=True)     # remove DC offset
    os.makedirs(cachedir, exist_ok=True)
    # np.savez appends .npz unless the name already ends in it, so the
    # temp name must too or os.replace looks for the wrong file
    tmp = cache[:-4] + ".tmp.npz"
    np.savez(tmp, slab=slab, meta=np.array(m, dtype=object))
    os.replace(tmp, cache)
    return slab, m


def clutter_centroid_hz(x, prf=PRF_HZ):
    """Mean Doppler from the lag-one autocorrelation phase."""
    return float(np.angle(np.mean(x[1:] * np.conj(x[:-1]))) * prf
                 / (2 * np.pi))


def selftest(root=None, cachedir=None):
    import glob
    root = root or os.environ.get(
        "IPIX_DATA_ROOT", r"G:/Actual_IPIX_Radar_Data/All_14p2_Files")
    cachedir = cachedir or os.path.join("data", "cache", "ipix")
    files = sorted(glob.glob(os.path.join(root, "*.[cC][dD][fF]")))
    print(f"{len(files)} IPIX files under {root}")
    slab, m = cell_slab(files[0], cachedir)
    print(f"  {m['file']}  {m['date']}  {m['site']}")
    print(f"  slab {slab.shape} {slab.dtype}  (sweep, cell, [HH,VV])")
    hh, vv = slab[:, NEEDED.index(0), 0], slab[:, NEEDED.index(0), 1]
    print(f"  CUT centroid  HH {clutter_centroid_hz(hh):+7.2f} Hz   "
          f"VV {clutter_centroid_hz(vv):+7.2f} Hz")
    print(f"  |rho(HH,VV)| = "
          f"{abs(np.vdot(hh, vv)) / np.sqrt(np.vdot(hh, hh).real * np.vdot(vv, vv).real):.3f}")
    return slab, m


if __name__ == "__main__":
    selftest()
