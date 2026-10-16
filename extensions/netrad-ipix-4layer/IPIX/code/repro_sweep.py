"""Reproduction sweep driver: registered v4 protocol + the DU-CCEBM arm.

Runs the FOLDER's registered protocol (sweep_ipix_v4.py, copied verbatim
into _work_repro/IPIX_Blind_Results/code) unmodified -- same 8
acquisitions (FILE_STRIDE=2), same 1500 H0 / 150 H1 windows, same SCR
levels, Doppler offsets, thresholds, checkpoint format -- with one added
blind arm: DU-CCEBM, unrolled from the folder's own ccebm_v3.py
(src/cerbm/du_ccebm_ipix.py).  The patch is one function wrap; nothing
in the protocol file is edited.

Checkpoints carry the suffix ``_v4duc.json`` in
_work_repro/results/checkpoints/ so no archived result anywhere can be
touched.  Merged results go to _work_repro/results/ipix_sweep_v4duc.json.

Usage:
    python _work_repro/experiments/repro_sweep.py --shard=0/3
    python _work_repro/experiments/repro_sweep.py --merge
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WROOT = os.path.normpath(os.path.join(_HERE, ".."))          # _work_repro
sys.path.insert(0, _HERE)

_FOLDER = os.path.join(_WROOT, "IPIX_Blind_Results", "code")
_DM = os.environ.get("DU_CCEBM_DATAMANIP_CODE", "")
_NR = os.environ.get("DU_CCEBM_NETRAD_CODE", "")
_SRC = os.path.normpath(os.path.join(_WROOT, "..", "src", "cerbm"))
for p in (_FOLDER, _DM, _NR, _SRC):
    if p and p not in sys.path:
        sys.path.insert(0, p)

RES = os.path.join(_WROOT, "results")
CKPT_DIR = os.path.join(RES, "checkpoints")
CKPT_SUFFIX = "_v4duc.json"

# DU-CCEBM arm config: same budget as the registered DU-cERBM arm
# (du_sweep_arm.DU_CFG): 4 unrolled layers, 1 epoch, warm start 6
# classical iterations, 200-candidate blind random search, statistic-
# aligned objective on the M=128 tail window.
from du_ccebm_ipix import DuCcebmConfig, du_ccebm_separate     # noqa: E402

DU_CFG = DuCcebmConfig(n_layers=4, n_epochs=1, warm_start=True,
                       warm_iters=6, mode="stat", M=128, lr=0.02,
                       n_random_search=200)


def load_sweep():
    """Import the folder sweep module by path, then patch in DU-CCEBM."""
    spec = importlib.util.spec_from_file_location(
        "sweep_ipix_v4", os.path.join(_FOLDER, "sweep_ipix_v4.py"))
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["sweep_ipix_v4"] = sweep
    spec.loader.exec_module(sweep)

    # Windows hardening: a transient sharing violation on os.replace
    # (search indexer / desktop app briefly holding the checkpoint) must
    # not kill a multi-hour shard.  Retry the final replace with backoff;
    # semantics identical, protocol file untouched.
    _base_atomic = sweep.atomic_json

    def _atomic_json_retry(path, obj, attempts=8):
        for k in range(attempts):
            try:
                _base_atomic(path, obj)
                return
            except PermissionError:
                if k == attempts - 1:
                    raise
                time.sleep(0.5 * (2 ** k))

    sweep.atomic_json = _atomic_json_retry

    base = sweep.separate

    def patched(a, b):
        out = dict(base(a, b))
        out["du_ccebm"] = du_ccebm_separate(np.vstack([a, b]), DU_CFG, M=128)
        return out

    sweep.separate = patched
    sweep.BLIND_ARMS = tuple(sweep.BLIND_ARMS) + ("du_ccebm",)
    return sweep


def main() -> None:
    sweep = load_sweep()
    os.makedirs(CKPT_DIR, exist_ok=True)

    shard, nshard = 0, 1
    for arg in sys.argv[1:]:
        if arg.startswith("--shard"):
            shard, nshard = (int(v) for v in arg.split("=")[1].split("/"))

    root = os.environ.get(
        "IPIX_DATA_ROOT", r"G:/Actual_IPIX_Radar_Data/All_14p2_Files")
    cachedir = os.environ.get(
        "IPIX_CACHE_DIR", os.path.normpath(
            os.path.join(_WROOT, "..", "data", "cache", "ipix")))
    os.makedirs(cachedir, exist_ok=True)

    import glob
    files = sorted(glob.glob(os.path.join(root, "*.[cC][dD][fF]")))
    all_files = list(files)[::sweep.FILE_STRIDE]
    files = all_files[shard::nshard]
    print(f"[duc shard {shard + 1}/{nshard}] {len(files)} of "
          f"{len(all_files)} files; arms {list(sweep.BLIND_ARMS)}",
          flush=True)

    lock_path = os.path.join(RES, f"repro_v4duc_{shard}of{nshard}.lock")
    with sweep.Lock(lock_path):
        for i, path in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(path))[0]
            ck = os.path.join(CKPT_DIR, stem + CKPT_SUFFIX)
            t0 = time.time()
            print(f"  [{i}/{len(files)}] {stem}", flush=True)
            try:
                sweep.run_file(path, cachedir, ck, verbose=True)
            except Exception:
                import traceback
                err = {"error": traceback.format_exc()}
                sweep.atomic_json(ck, err)
                print(f"    ERROR on {stem} (checkpointed)", flush=True)
            print(f"    done in {time.time() - t0:.0f}s", flush=True)


def merge_files() -> None:
    sweep = load_sweep()
    import glob as _g
    cks = sorted(_g.glob(os.path.join(CKPT_DIR, "*" + CKPT_SUFFIX)))
    out = {}
    for ck in cks:
        stem = os.path.splitext(os.path.basename(ck))[0].replace(
            CKPT_SUFFIX, "")
        try:
            with open(ck, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            continue
        if st.get("complete"):
            out[stem] = {k: st[k] for k in
                         ("levels", "meta", "target_doppler_hz",
                          "clutter_centroid_hz", "K_local_by_M", "thr")
                         if k in st}
        elif "error" in st:
            out[stem] = {"error": st["error"]}
    done = sum(1 for v in out.values() if "error" not in v)
    payload = {"provenance": {
        "script": "_work_repro/experiments/repro_sweep.py",
        "base_protocol": "IPIX_Blind_Results/code/sweep_ipix_v4.py",
        "tail": sweep.TAIL, "decision_windows": sweep.MS,
        "scr_levels_db": sweep.SCR_LEVELS, "n_h0": sweep.N_H0,
        "n_h1": sweep.N_H1, "pfa_nominal": sweep.PFA,
        "prf_hz": sweep.PRF_HZ, "deltas_hz": list(sweep.DELTAS_HZ),
        "file_stride": sweep.FILE_STRIDE,
        "blind_arms": list(sweep.BLIND_ARMS),
        "du_ccebm_config": {k: str(v) for k, v in vars(DU_CFG).items()
                            if not k.startswith("_")
                            and not hasattr(v, "__call__")},
        "n_files": done},
        "files": out}
    q = os.path.join(RES, "ipix_sweep_v4duc.json")
    sweep.atomic_json(q, payload)
    print(f"  merged {done}/{len(cks)} complete acquisitions -> {q}",
          flush=True)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_files()
    else:
        main()
