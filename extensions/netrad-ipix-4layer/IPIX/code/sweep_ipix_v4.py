"""
IPIX diagnostic sweep, v4 -- the CCEBM implementation question, with a
CCEBM that has been checked against ground truth first.

WHY v4 REPLACES v3
------------------
v3 died: two of its three shards lost every acquisition to a
LinAlgError, and the one that survived was computing the wrong thing.
Three defects, all in the arm that was supposed to test the paraunitary
hypothesis, all found after the run:

 1. THE PROJECTION NEVER CONVERGED. The Douglas iteration left a
    constraint residual of 0.2-0.5 at the eight sweeps the sweep could
    afford, and two of five random starts sat on a non-paraunitary fixed
    point for ever. The "paraunitary" arm was imposing nothing.
    Replaced by pu_lattice.py: Vaidyanathan's factorisation, which is
    paraunitary to 2e-15 by construction for ANY parameters, so there is
    nothing to project and nothing to diverge.

 2. THE GRADIENT COULD OVERFLOW. A widely linear filter that nearly
    annihilates its own source drives a standard deviation towards zero,
    and the gradient carries 1/s and 1/s**2. Once that overflowed to
    inf, normalisation turned it into nan and np.linalg.eig raised. That
    is what killed the shards. cost_of now returns inf for any
    non-finite or degenerate iterate, so the line search simply rejects
    the step -- the machinery for that was already there and was being
    bypassed by an exception.

 3. THE COMPARISON WAS SCORED WRONG, BY ME. Convolutive ICA recovers
    sources up to a DELAY. Scoring correlation at lag zero only, the
    lattice arm looked as though it recovered one source and lost the
    other (0.949 / 0.041). Scored over the delay ambiguity it recovers
    both (0.949 at lag 1, 0.930 at lag 3). Every "cERBM beats CCEBM"
    number computed that way was measuring my readout, not the
    algorithms. The detection statistic in this sweep was never affected
    -- it reads coherence with a Doppler steering vector, whose
    magnitude a delay does not change -- but the bench comparison was.

WHAT THE BENCH SAYS NOW (12 synthetic convolutive mixtures, ground truth
known, scored over the delay ambiguity, median |corr|):

    arm                            tone   bpsk   both>0.7
    ccebm_dp  real-AR, orthogonal  0.429  0.441    0.00     <- the v2 arm
    ccebm_wl  Eq (13), orthogonal  0.873  0.798    0.50
    ccebm_pu  Eq (13), lattice     0.725  0.697    0.33
    ccebm_pu  x3 restarts          0.726  0.810    0.58
    cerbm                          0.689  0.738    0.33

So the arm this sweep has been calling CCEBM since v2 -- ccebm_dp -- is
far and away the worst of them, and the CCEBM of Eq (13) is ahead of
cERBM rather than behind it. That is the direction the hand experiments
reported and the direction v2 and v3 contradicted. All four arms are
carried here so the sweep says which of them the claim depends on.

RESTARTS ARE ONLY LEGITIMATE ON THE pu ARM. Over 20 seeds on the same
known mixture, cost correlates with separation at -0.39 under the exact
paraunitary constraint (lower cost, better separation -- the right sign)
and at +0.50 without it. Selecting the lowest-cost restart is an
unsupervised choice that never looks at the target, but without the
constraint it would reliably select the WORST solution.

TWO SAVINGS THAT COST NOTHING
  * Under H0 no target is injected, so neither the separators nor the
    covariance inverses depend on the target Doppler -- only the readout
    does. Both offsets therefore share one pass, halving the H0 cost.
  * M = 512 is dropped. The analysis marks it CONFOUNDED anyway (K = 2M
    secondary samples exceed the clutter's stationarity interval), it is
    the most expensive window by far, and v2 already carries it for the
    headline. This run is a diagnostic about CCEBM variants.

CRASH RESILIENCE, unchanged from v3: one atomically-replaced checkpoint
per acquisition, rewritten after every H0 block and every SCR level; a
pid lock that detects its own staleness after a hard power-off; a cell
cache that is rebuilt rather than trusted if it fails to load.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "Data_manipulation", "code")))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "NetRAD_Data_ExplanationCode", "netrad", "code")))

from ipix_reader import (                                    # noqa: E402
    PRF_HZ, REF_OFFSETS, NEEDED, cell_slab, clutter_centroid_hz)
from cerbm import cerbm                                      # noqa: E402
from datamanip_tail_arms import (                             # noqa: E402
    PFA, steering, cov_from_blocks, tail_filter, apply_filter)
from stfica_datamanip_detector import (                       # noqa: E402
    amf_glrt, kelly_glrt, anmf, ca_cfar, stfica, L_FILT, SCAN_OVERSAMPLE)
from ccebm_datamanip_detector import ccebm_v2                 # noqa: E402
from ccebm_v3 import ccebm_v3                                 # noqa: E402
from sweep_decision_window import scan_via_fft                # noqa: E402

TAIL = 5000
MS = [10, 32, 128]
M_MAX = max(MS)
SCR_LEVELS = [-30, -25, -20, -15, -10, -5, 0, 5]
N_H0 = 1500
N_H1 = 150
FILE_STRIDE = 2          # 8 of the 16, both campaigns; fixed before the run
DELTAS_HZ = (21.0, 60.0)
CERBM_ITER = 25
WL_KW = dict(L=6, K=4, n_iter=15, paraunitary_on=False,
             whiten_mode="spatial")
PU_KW = dict(L=6, K=4, n_iter=15, paraunitary_on=True,
             whiten_mode="spatial", n_restart=3)
BLIND_ARMS = ("stfica", "ccebm_dp", "ccebm_wl", "ccebm_pu", "cerbm")
TGT_POL = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2.0)
CKPT_SUFFIX = "_v4.json"


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Lock:
    """Single-instance lock that survives a hard power-off: a lock whose
    pid is no longer alive is stale and gets cleared."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    pid = int(json.load(f)["pid"])
                alive = os.system(f'tasklist /FI "PID eq {pid}" '
                                  f'| findstr /I "{pid}" >nul 2>&1') == 0
            except Exception:
                alive = False
            if alive:
                raise SystemExit(f"another run holds {self.path}; exiting")
            os.remove(self.path)
        atomic_json(self.path, {"pid": os.getpid(), "started": time.time()})
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def separate(a, b):
    """Every blind arm, on one window. Returns {arm: sources}."""
    pair = np.vstack([a, b])
    return {
        "stfica": stfica(np.real(a), np.imag(a), L_FILT),
        "ccebm_dp": ccebm_v2(pair, L_FILT)[0],
        "ccebm_wl": ccebm_v3(pair, **WL_KW)[0],
        "ccebm_pu": ccebm_v3(pair, **PU_KW)[0],
        "cerbm": cerbm(pair, n_iter=CERBM_ITER)[0],
    }


def run_file(path, cachedir, ckpt_path, verbose=True):
    state = {}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    if state.get("complete"):
        return state

    slab, m = cell_slab(path, cachedir)
    ci = NEEDED.index(0)
    hh = slab[:, ci, 0].astype(np.complex128)
    vv = slab[:, ci, 1].astype(np.complex128)
    refs = [slab[:, NEEDED.index(o), 0].astype(np.complex128)
            for o in REF_OFFSETS]
    centroid = clutter_centroid_hz(hh)
    p_clut = float(np.mean(np.abs(hh) ** 2))
    n_sweep = len(hh)
    starts_h0 = np.linspace(TAIL, n_sweep - M_MAX - 1, N_H0).astype(int)
    starts_h1 = np.linspace(TAIL, n_sweep - M_MAX - 1, N_H1).astype(int)

    FD, FX = {}, {}
    for dlt in DELTAS_HZ:
        f = (centroid + dlt + PRF_HZ / 2) % PRF_HZ - PRF_HZ / 2
        FD[dlt] = f
        FX[dlt] = {M: {"v": steering(M, f),
                       "vr": np.exp(-2j * np.pi * f * np.arange(M) / PRF_HZ),
                       "n_scan": M * SCAN_OVERSAMPLE} for M in MS}

    def secondary(s, M, span_limit=None):
        K = 2 * M
        per = int(np.ceil(K / len(refs)))
        if span_limit is not None:
            per = max(1, min(per, span_limit // M))
        half = per * M // 2
        lo = max(0, s - half)
        sec = np.concatenate([r[lo:lo + per * M] for r in refs])
        return np.linalg.inv(cov_from_blocks(sec, M)), len(sec) // M

    def stats(s, scr, dlts):
        """One window, read out at every Doppler offset in dlts.

        With scr None nothing is injected, so the separators and the
        covariance inverses are independent of the target Doppler and are
        computed ONCE for all offsets; only the steering vectors differ.
        With a target present each offset needs its own window, so dlts
        carries a single entry.
        """
        a0 = hh[s - TAIL:s + M_MAX]
        b0 = vv[s - TAIL:s + M_MAX]
        out = {}
        blind_clean = separate(a0.copy(), b0.copy()) if scr is None else None
        inv = {M: secondary(s, M) for M in MS} if scr is None else None
        for dlt in dlts:
            f_d, fx = FD[dlt], FX[dlt]
            a, b = a0.copy(), b0.copy()
            if scr is None:
                blind = blind_clean
            else:
                amp = np.sqrt(p_clut * 10 ** (scr / 10.0))
                tone = amp * np.exp(2j * np.pi * f_d
                                    * np.arange(len(a)) / PRF_HZ)
                a = a + TGT_POL[0] * tone
                b = b + TGT_POL[1] * tone
                blind = separate(a, b)
            d = {}
            for M in MS:
                F = fx[M]
                y = a[-M:]
                Rinv, Kact = (inv[M] if inv is not None
                              else secondary(s, M))
                refw = [r[s + M_MAX - M:s + M_MAX] for r in refs]
                d["ca_cfar_M%d" % M] = ca_cfar(y, refw, F["v"])
                d["anmf_M%d" % M] = anmf(y, Rinv, F["v"])
                a_sc, k_sc = scan_via_fft(y, Rinv, M, F["n_scan"], Kact)
                d["anmf_scan_M%d" % M] = a_sc
                d["kelly_scan_M%d" % M] = k_sc
                d["amf_glrt_M%d" % M] = amf_glrt(y, Rinv, F["v"])
                d["kelly_glrt_M%d" % M] = kelly_glrt(y, Rinv, F["v"], Kact)
                Rl, Kl = secondary(s, M, span_limit=TAIL)
                d["anmf_local_M%d" % M] = anmf(y, Rl, F["v"])
                d["kelly_local_M%d" % M] = kelly_glrt(y, Rl, F["v"], Kl)
                d["_K_local_M%d" % M] = float(Kl)
                w = tail_filter(a[:-M_MAX], L_FILT)
                r = apply_filter(np.concatenate(
                    [a[-M_MAX - L_FILT - 1:-M_MAX], y]), w)
                d["wiener_M%d" % M] = float(
                    np.abs(steering(len(r), f_d).conj() @ r) ** 2)
                for nm, Sx in blind.items():
                    best = 0.0
                    for row in Sx[:, -M:]:
                        e = float(np.abs(np.vdot(row, row)).real)
                        if e > 0:
                            best = max(best,
                                       float(np.abs(row @ F["vr"]) ** 2 / e))
                    d["%s_M%d" % (nm, M)] = best
            out["d%g" % dlt] = d
        return out

    state.setdefault("levels", {})
    state.setdefault("thr", {})
    if not all(("d%g" % d) in state["thr"] for d in DELTAS_HZ):
        acc = None
        done = int(state.get("h0_done", 0))
        t0 = time.time()
        while done < N_H0:
            rows = [stats(x, None, DELTAS_HZ)
                    for x in starts_h0[done:done + 250]]
            if acc is None:
                prev = state.get("h0_acc", {})
                acc = {dk: {k: list(prev.get(dk, {}).get(k, []))
                            for k in rows[0][dk] if not k.startswith("_")}
                       for dk in rows[0]}
                state["K_local_by_M"] = {
                    k: rows[0][list(rows[0])[0]][k]
                    for k in rows[0][list(rows[0])[0]]
                    if k.startswith("_K_local")}
            for r in rows:
                for dk in acc:
                    for k in acc[dk]:
                        acc[dk][k].append(r[dk][k])
            done += len(rows)
            state["h0_acc"], state["h0_done"] = acc, done
            state["clutter_centroid_hz"] = centroid
            state["target_doppler_hz"] = {("d%g" % d): FD[d]
                                          for d in DELTAS_HZ}
            state["meta"] = m
            atomic_json(ckpt_path, state)
            if verbose:
                print("      H0 %d/%d (%.0fs)"
                      % (done, N_H0, time.time() - t0), flush=True)
        for dk in acc:
            state["thr"][dk] = {k: float(np.quantile(v, 1 - PFA))
                                for k, v in acc[dk].items()}
        state.pop("h0_acc", None)
        atomic_json(ckpt_path, state)

    for dlt in DELTAS_HZ:
        dk = "d%g" % dlt
        thr = state["thr"][dk]
        for scr in SCR_LEVELS:
            key = "%s|%d" % (dk, scr)
            if key in state["levels"]:
                continue
            t0 = time.time()
            rows = [stats(x, scr, (dlt,))[dk] for x in starts_h1]
            state["levels"][key] = {
                k: float(np.mean([r[k] > thr[k] for r in rows]))
                for k in thr}
            atomic_json(ckpt_path, state)
            if verbose:
                L = state["levels"][key]
                print("      %s SCR %4d (%.0fs)  M128: kelly %.2f  dp %.2f "
                      " wl %.2f  pu %.2f  cerbm %.2f"
                      % (dk, scr, time.time() - t0, L["kelly_glrt_M128"],
                         L["ccebm_dp_M128"], L["ccebm_wl_M128"],
                         L["ccebm_pu_M128"], L["cerbm_M128"]), flush=True)
    state["complete"] = True
    atomic_json(ckpt_path, state)
    return state


def _is_complete(ck):
    try:
        with open(ck, encoding="utf-8") as f:
            return bool(json.load(f).get("complete"))
    except Exception:
        return False


def merge(ckptdir, resdir, files):
    out = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        ck = os.path.join(ckptdir, stem + CKPT_SUFFIX)
        if not os.path.exists(ck):
            continue
        try:
            with open(ck, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            continue
        if st.get("complete"):
            out[os.path.basename(path)] = {
                k: st[k] for k in ("levels", "meta", "target_doppler_hz",
                                   "clutter_centroid_hz", "K_local_by_M",
                                   "thr") if k in st}
        elif "error" in st:
            out[os.path.basename(path)] = {"error": st["error"]}
    done = sum(1 for v in out.values() if "error" not in v)
    payload = {"provenance": {
        "script": "IPIX_Blind_Results/code/sweep_ipix_v4.py",
        "tail": TAIL, "decision_windows": MS,
        "scr_levels_db": SCR_LEVELS, "n_h0": N_H0, "n_h1": N_H1,
        "pfa_nominal": PFA, "prf_hz": PRF_HZ,
        "deltas_hz": list(DELTAS_HZ), "file_stride": FILE_STRIDE,
        "blind_arms": list(BLIND_ARMS),
        "secondary_rule": "K = 2M, Reed-Mallett-Brennan",
        "ccebm_wl_settings": {k: str(v) for k, v in WL_KW.items()},
        "ccebm_pu_settings": {k: str(v) for k, v in PU_KW.items()},
        "cerbm_iter": CERBM_ITER, "n_files": done,
        "target_polarisation": "equal HH/VV magnitude, 45 deg phase"},
        "files": out}
    q = os.path.join(resdir, "ipix_sweep_v4.json")
    atomic_json(q, payload)
    print(f"  merged {done}/{len(files)} complete acquisitions -> {q}",
          flush=True)


def main():
    shard, nshard = 0, 1
    for arg in sys.argv[1:]:
        if arg.startswith("--shard"):
            shard, nshard = (int(v) for v in arg.split("=")[1].split("/"))
    root = os.environ.get(
        "IPIX_DATA_ROOT", r"G:/Actual_IPIX_Radar_Data/All_14p2_Files")
    resdir = os.path.normpath(os.path.join(_HERE, "..", "results"))
    ckptdir = os.path.join(resdir, "checkpoints")
    cachedir = os.environ.get(
        "IPIX_CACHE_DIR",
        os.path.normpath(os.path.join(_HERE, "..", "..", "data", "cache",
                                      "ipix")))
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(root, "*.[cC][dD][fF]")))
    all_files = list(files)[::FILE_STRIDE]
    files = all_files[shard::nshard]
    print(f"[shard {shard + 1}/{nshard}] {len(files)} of {len(all_files)} "
          f"files")
    print(f"IPIX v4: M {MS}, tail {TAIL}, SCR {SCR_LEVELS[0]}.."
          f"{SCR_LEVELS[-1]}, K=2M, {N_H0} H0 + {N_H1} H1, arms "
          f"{list(BLIND_ARMS)}", flush=True)

    with Lock(os.path.join(resdir, f"sweep_v4_{shard}of{nshard}.lock")):
        t_all = time.time()
        for i, path in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(path))[0]
            ck = os.path.join(ckptdir, stem + CKPT_SUFFIX)
            done_levels = 0
            if os.path.exists(ck):
                try:
                    with open(ck, encoding="utf-8") as f:
                        st = json.load(f)
                    if st.get("complete"):
                        print(f"  [{i}/{len(files)}] {stem} already done",
                              flush=True)
                        continue
                    done_levels = len(st.get("levels", {}))
                except Exception:
                    pass
            print(f"  [{i}/{len(files)}] {stem}"
                  + (f"  (resuming at {done_levels}/"
                     f"{len(SCR_LEVELS) * len(DELTAS_HZ)} levels)"
                     if done_levels else ""), flush=True)
            t0 = time.time()
            try:
                run_file(path, cachedir, ck)
                print(f"      file done in {time.time() - t0:.0f}s "
                      f"({(time.time() - t_all) / 60:.0f} min elapsed)",
                      flush=True)
            except Exception:
                err = traceback.format_exc()[-1500:]
                atomic_json(ck, {"error": err})
                print(f"      FAILED\n{err}", flush=True)
        merge(ckptdir, resdir, all_files)
    # Exit non-zero unless every file in THIS shard's slice really
    # finished. v3's launcher retried on the string "sweep finished",
    # which the script prints even when every acquisition has failed --
    # so two shards reported success having produced nothing. The
    # launcher now retries on the exit code instead.
    bad = [os.path.splitext(os.path.basename(p))[0] for p in files
           if not _is_complete(os.path.join(
               ckptdir, os.path.splitext(os.path.basename(p))[0]
               + CKPT_SUFFIX))]
    if bad:
        print(f"sweep INCOMPLETE: {len(bad)} of {len(files)} unfinished "
              f"({', '.join(bad[:4])})", flush=True)
        return 1
    print("sweep finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
