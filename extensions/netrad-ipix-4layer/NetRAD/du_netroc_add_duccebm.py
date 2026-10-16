"""DU-CCEBM arm execution on the NetRAD Path-A protocol (node N3).

New paradigm arm per the user's request: the paper's DU-CCEBM (Ghahramani,
``du_ccebm_signal_processing-4.pdf`` -- deep-unfolded convolutive
complex-valued CCEBM, structure-preserving trainable layers) is RUN
on exactly the same windows / offsets/SCRs as the registered
``du_roc_netrad.py`` run, and its per-window Doppler-steered statistic is
appended to each NetRAD checkpoint.  Existing arm statistics are NEVER
touched -- the new arm is written back to the checkpoint in an extra key
``du_ccebm`` (with exactly the same per-window ordering), so the
registered analysis gates and the existing 10-arm claims stay intact.

The used registered protocol constants mirror ``du_roc_netrad.py``
byte-for-byte:
  TAIL 5000, M 128, N_H0 400, N_H1 5x80, SCR -25..-5 step 5 (dB),
  offset +21 Hz about the measured clutter centroid, seed 20260907.

Checkpoints give full reproducibility; this is exactly how
``netrad_roc_*_v1.json`` was always produced.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_FOLDER = os.path.join(_ROOT, "IPIX_Blind_Results", "code")
for _p in (_FOLDER,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu  # noqa: E402
_spec_nr = _ilu.spec_from_file_location(
    "netrad_reader", os.path.join(_FOLDER, "netrad_reader.py"))
nr = _ilu.module_from_spec(_spec_nr)
sys.modules["netrad_reader"] = nr
_spec_nr.loader.exec_module(nr)

_spec_du = _ilu.spec_from_file_location(
    "du_ccebm_ipix", os.path.join(_ROOT, "src", "cerbm", "du_ccebm_ipix.py"))
du = _ilu.module_from_spec(_spec_du)
sys.modules["du_ccebm_ipix"] = du
_spec_du.loader.exec_module(du)

ARM = "du_ccebm"
CKPT_SUFFIX = "_v1.json"
FOLDER_RES = os.path.join(_ROOT, "IPIX_Blind_Results", "results")
CKPTDIR = os.path.join(FOLDER_RES, "checkpoints_du")

# registered protocol constants (byte-identical to du_roc_netrad.py)
TAIL = 5000
M = 128
M_MAX = M
N_H0 = 400
N_H1 = 80
SCR_LEVELS = np.array([-25.0, -20.0, -15.0, -10.0, -5.0])
OFFSET_HZ = 21.0
SEED = 20260907

CFG_DU = dict(n_layers=4, warm_iters=6, n_epochs=1, n_random_search=200,
              M=M, L=6, K=4, lr=0.02, seed=SEED)


def atomic_json(path, obj):
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_one(ts: str, cachedir: str) -> None:
    ck = os.path.join(CKPTDIR, "netrad_roc_" + ts + CKPT_SUFFIX)
    if not os.path.exists(ck):
        sys.exit(f"missing checkpoint {ck} -- the registered run must be "
                 "complete before the DU-CCEBM arm can be appended")
    with open(ck, encoding="utf-8") as f:
        state = json.load(f)
    if not state.get("complete"):
        sys.exit(f"checkpoint {ck} not complete")
    if ARM in state["h0"]:
        if len(state["h0"][ARM]) == N_H0 and \
                all(len(state["levels"][k][ARM]) == N_H1
                    for k in state["levels"]):
            print(f"  {ts} du_ccebm already present", flush=True)
            return
    cells, meta = nr.cell_slab(nr.file_for(ts), cachedir, ts)
    if isinstance(meta, str):
        meta = json.loads(meta)
    cells = cells.astype(np.complex128)
    cut_i = cells.shape[1] // 2
    cut = cells[:, cut_i]
    sec_cells = [c for c in range(cells.shape[1]) if c != cut_i]
    sec_cells.sort(key=lambda c: (abs(c - cut_i), c))
    refs = [cells[:, c] for c in sec_cells[:12]]

    centroid = nr.clutter_centroid_hz(cut)
    n_sweep = len(cut)
    starts_h0 = np.linspace(TAIL, n_sweep - M_MAX - 1, N_H0).astype(int)
    starts_h1 = np.linspace(TAIL, n_sweep - M_MAX - 1, N_H1).astype(int)

    f_d = (centroid + OFFSET_HZ + nr.PRF_HZ / 2) % nr.PRF_HZ \
        - nr.PRF_HZ / 2
    vr = np.exp(-2j * np.pi * f_d * np.arange(M) / nr.PRF_HZ)

    p_clut = float(np.mean(np.abs(cut) ** 2))
    TGT_POL = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2.0)

    def mk_window(s, amp=None):
        a = cut[s - TAIL:s + M_MAX].copy()
        b = refs[0][s - TAIL:s + M_MAX].copy()
        if amp is not None:
            tone = amp * np.exp(
                2j * np.pi * f_d
                * np.arange(s + M_MAX - (s - TAIL)) / nr.PRF_HZ)
            a = a + TGT_POL[0] * tone
            b = b + TGT_POL[1] * tone
        return np.vstack([a, b])

    def stat_window(pair):
        Wv = du.du_ccebm_separate(
            pair, du.DuCcebmConfig(**CFG_DU), M=M)
        best = 0.0
        for row in Wv[:, -M:]:
            e = float(abs(np.vdot(row, row)).real)
            if e > 0:
                best = max(best, float(abs(row @ vr) ** 2 / e))
        return best

    # allocate the per-arm arrays if missing
    state["h0"].setdefault(ARM, [])
    for scr in SCR_LEVELS:
        state["levels"].setdefault("%g" % scr, {}).setdefault(ARM, [])

    print(f"{ts}: appending DU-CCEBM stats (K={len(CFG_DU)} cfg)",
          flush=True)
    t0 = time.time()
    todo_h0 = [k for k in range(N_H0) if len(state["h0"][ARM]) <= k]
    for i, k in enumerate(todo_h0):
        s = starts_h0[k]
        st = stat_window(mk_window(s))
        state["h0"][ARM].append(st)
        if (i + 1) % 10 == 0 or i == len(todo_h0) - 1:
            atomic_json(ck, state)
            print(f"  {ts} H0 {len(state['h0'][ARM])}/{N_H0}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    for scr in SCR_LEVELS:
        key = "%g" % scr
        acc = state["levels"][key]
        todo_h1 = [i for i in range(N_H1)
                   if len(acc[ARM]) <= i]
        if not todo_h1:
            continue
        t1 = time.time()
        amp = float(np.sqrt(p_clut * 10 ** (scr / 10.0)))
        for i in todo_h1:
            s = starts_h1[i]
            st = stat_window(mk_window(s, amp))
            acc[ARM].append(st)
            atomic_json(ck, state)
            print(f"  {ts} SCR {scr:+g} H1 {len(acc[ARM])}/{N_H1}"
                  f" ({time.time() - t1:.0f}s)", flush=True)
    state["complete"] = True
    atomic_json(ck, state)
    print(f"  {ts} DU-CCEBM COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cachedir = os.path.join(_ROOT, "data", "cache", "netrad")
    os.makedirs(cachedir, exist_ok=True)
    ts_all = nr.timestamps()
    if args.limit:
        ts_all = ts_all[:args.limit]
    for ts in ts_all[args.shard::args.nshard]:
        run_one(ts, cachedir)
    print("du_ccebm collection done", flush=True)


if __name__ == "__main__":
    main()
