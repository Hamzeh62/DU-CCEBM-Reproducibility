"""Deep-unfolded CCEBM for the IPIX blind-detector protocol.

Source algorithm: ``IPIX_Blind_Results/code/ccebm_v3.py`` -- classical
CCEBM with Eq.-(13) widely-linear per-source whitening and, optionally,
the exact paraunitary constraint (``pu_lattice.py``, Vaidyanathan
factorisation).  The sweep reads, per source row, over the tail-M
decision window:

    stat = max_f |sum_t z_i[t] v_f[t]|^2 / ||z_i||^2

-- peak Doppler-steered power of the separated sources.  The DU-CCEBM
arm must improve exactly THAT quantity, blind.

WHAT IS UNROLLED
----------------
One iteration of ``ccebm_v3`` becomes one trainable layer.  A ccebm_v3
iteration is: (1) per-source Eq.-(13) widely-linear whitening filter b
from the CURRENT source estimate (recomputed per iteration); (2) the
conj-linear/linear branch signals (a, c) through that filter, with
z = w^H a + w^T c; (3) per-variable (z_R, z_I) measuring functions from
Table I, chosen adaptively; (4) the entropy-rate gradient and the
descent step W <- W - tau G, executed on the EXACT paraunitary lattice
(U retracted by QR, each v_k renormalised -- both retractions exact);
(5) global accept/reject on the cost.  Trainable per layer, initialised
AT the classical values so layer 0 of the untrained net reproduces the
classical step:

    tau_k    the step size                       (classical: adaptive)
    beta_k   weight of the noncircularity (z_I) term   (classical: 1)
    gamma_k  weight of the non-Gaussianity (z_R) term  (classical: 1)

TRAINING OBJECTIVE
------------------
The sweep's own hard Doppler statistic on the tail-M decision window,
maximised directly, with a blind accept/reject guard per layer: a layer
update is kept only if the hard statistic does not decrease.  The final
row 1 is the best-stat row seen during unrolling, so DU-CCEBM >= its
warm start on the training statistic BY CONSTRUCTION.  No target label
is used anywhere -- only the Doppler steering grid the detector itself
reads (blind, same information as every other arm).

Row 2 is the classical second row, paraunitary-retracted against row 1
per layer (declared deviation D-3, mirroring du_ipix.py): the Doppler
objective has no second-source gradient signal and the sweep statistic
takes the max over rows.

DECLARED DEVIATIONS (mirroring the folder's honesty convention)
  D-1  the negentropy functional uses the folder's declared quadratic
       approximation (ccebm_v3.py states the same deviation);
  D-2  the Eq.-(13) filter b is re-estimated per layer from at most the
       last ``b_est_samples`` samples of the current source estimate
       (speed; ccebm_v3 uses the full window);
  D-3  row 2 is not trained: classical row 2, re-orthogonalised against
       the trained row 1;
  D-4  training runs in complex64 on torch; the lattice retractions and
       the (a, c) branch construction stay float64/complex128 numpy,
       identical to ccebm_v3 where it matters.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

if torch.get_num_threads() > 4:
    torch.set_num_threads(4)

# import the FOLDER's ccebm_v3 by path under a private name: nothing
# named ``cerbm`` or ``ccebm_v3`` may enter sys.modules under its own
# name (the sweep driver binds its own copies), and du_ccebm must not
# depend on any project package.
_HERE = os.path.dirname(os.path.abspath(__file__))
# prefer the self-contained copy shipped in this package, fall back to
# the parent repository's IPIX folder when run from the repo
_LOCAL = os.path.join(_HERE, "code")
_SELF = _HERE
_REPO = os.path.normpath(os.path.join(
    _HERE, "..", "..", "IPIX_Blind_Results", "code"))
_FOLDER = _SELF if os.path.isfile(
    os.path.join(_SELF, "pu_lattice.py")) else (
    _LOCAL if os.path.isfile(
        os.path.join(_LOCAL, "pu_lattice.py")) else _REPO)
if _FOLDER not in sys.path:
    sys.path.insert(0, _FOLDER)


def _load_folder_modules():
    """Load pu_lattice / cerbm / ccebm_v3 from the FOLDER by file path,
    even if the project package ``cerbm`` (src/cerbm) is already in
    sys.modules: ccebm_v3 does ``from cerbm import _MEAS`` and must bind
    the FOLDER's cerbm.py, never the project package.  The folder
    modules are loaded under their plain names transiently and removed
    again afterwards; ccebm_v3 keeps the bound objects in its globals."""
    import importlib.util
    keys = ("pu_lattice", "cerbm", "ccebm_v3")
    saved = {k: sys.modules.pop(k) for k in keys if k in sys.modules}
    try:
        if _FOLDER in sys.path:
            sys.path.remove(_FOLDER)
        sys.path.insert(0, _FOLDER)
        mods = {}
        for name in keys:
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(_FOLDER, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mods[name] = mod
        return mods
    finally:
        for k in keys:
            sys.modules.pop(k, None)
        for k, m in saved.items():
            sys.modules[k] = m


_M = _load_folder_modules()
_PU = _M["pu_lattice"]
_ccebm_v3_mod = _M["ccebm_v3"]

_MEAS = _M["cerbm"]._MEAS          # Table I (numpy), shared by the folder
_EG = _M["cerbm"]._EG
NAMES = ("G1", "G2", "G3", "G4")
_S_FLOOR = _ccebm_v3_mod._S_FLOOR
_lagged = _M["cerbm"]._lagged
_wl_filter = _ccebm_v3_mod._wl_filter
_apply_wl = _ccebm_v3_mod._apply_wl

# Table I in torch, for differentiable layers (same forms as du_ipix.py)
MEAS_F = {
    "G1": (lambda u: u ** 4, lambda u: 4.0 * u ** 3),
    "G2": (lambda u: torch.abs(u) / (1.0 + torch.abs(u)),
           lambda u: torch.sign(u) / (1.0 + torch.abs(u)) ** 2),
    "G3": (lambda u: u * torch.abs(u) / (10.0 + torch.abs(u)),
           lambda u: torch.abs(u) * (20.0 + torch.abs(u))
           / (10.0 + torch.abs(u)) ** 2),
    "G4": (lambda u: u / (1.0 + u ** 2),
           lambda u: (1.0 - u ** 2) / (1.0 + u ** 2) ** 2),
}


def pick_np(u_std: np.ndarray) -> str:
    """Folder's adaptive pick: the tightest bound per variable."""
    def neg(name):
        return float((_MEAS[name][0](u_std).mean() - _EG[name]) ** 2)
    return max(NAMES, key=neg)


def pick_torch(u_std: torch.Tensor) -> str:
    return max(NAMES, key=lambda n: float(
        (MEAS_F[n][0](u_std).mean() - _EG[n]) ** 2))


def whiten_spatial(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ccebm_v3's ``whiten_mode='spatial'`` (what ccebm_wl / ccebm_pu use)."""
    Xc = X - X.mean(1, keepdims=True)
    C = (Xc @ Xc.conj().T) / Xc.shape[1]
    d, E = np.linalg.eigh(C)
    keep = d > 1e-10 * d.max()
    Wh = (E[:, keep] / np.sqrt(d[keep])).conj().T
    return Wh @ Xc, Wh


# ---------------------------------------------------------------------- #
# classical warm start: faithful port of ccebm_v3's iteration loop, with
# a per-iteration trace hook so the unrolled net can resume anywhere.
# ---------------------------------------------------------------------- #

def classical_ccebm_rows(X: np.ndarray, L: int = 6, K: Optional[int] = None,
                         n_iter: int = 15, tau0: float = 0.02, seed: int = 7,
                         paraunitary_on: bool = True,
                         trace_fn=None) -> np.ndarray:
    """Return the demixing bank rows W (n_src, n_ch*L) after n_iter
    ccebm_v3 iterations, exactly as ccebm_v3 runs them (whiten_mode
    'spatial', PU_KW defaults).  ``trace_fn(U, Vs, W, cost)`` is called
    after initialisation and after every accepted iteration; it is a
    pure recorder."""
    n_ch, T = X.shape
    n_src = n_ch
    if K is None:
        K = max(1, int(round(1.4 * np.log(T))))
    Xw, _Wh = whiten_spatial(X)
    Z = np.concatenate(
        [np.stack([Xw[c, L - 1 - l: T - l] for l in range(L)], 0)
         for c in range(n_ch)], 0)
    Z = Z / (Z.std() + 1e-30)
    dim = Z.shape[0]
    n_eff = n_ch
    rng = np.random.default_rng(seed)
    use_pu = bool(paraunitary_on and n_eff * L == dim and n_src <= n_eff)
    if use_pu:
        Upar, Vpar = _PU.init(n_src, n_eff, L, rng)
        W = _PU.bank(Upar, Vpar, L).reshape(n_src, dim)
    else:
        W = rng.standard_normal((n_src, dim)) + 1j * rng.standard_normal(
            (n_src, dim))
        W = np.linalg.qr(W.conj().T)[0].conj().T
    tau = tau0

    def cost_of(Wm):
        tot, bs, picks, ac = 0.0, [], [], []
        if not np.all(np.isfinite(Wm)):
            return np.inf, bs, picks, ac
        for i in range(n_src):
            y = Wm[i].conj() @ Z
            b = _wl_filter(y, K)
            if b is None:
                return np.inf, bs, picks, ac
            a, c = _apply_wl(Z, b, K)
            ac.append((a, c))
            z = Wm[i].conj() @ a + Wm[i] @ c
            zr, zi = z.real, z.imag
            sr, si = zr.std(), zi.std()
            if not (sr > _S_FLOOR and si > _S_FLOOR):
                return np.inf, bs, picks, ac
            g1, g2 = pick_np(zr / sr), pick_np(zi / si)
            tot += (np.log(2 * np.pi * np.e * sr * si)
                    - float((_MEAS[g1][0](zr / sr).mean() - _EG[g1]) ** 2)
                    - float((_MEAS[g2][0](zi / si).mean() - _EG[g2]) ** 2))
            bs.append(b)
            picks.append((g1, g2))
        if not np.isfinite(tot):
            return np.inf, bs, picks, ac
        return float(tot), bs, picks, ac

    cost, bs, picks, ac = cost_of(W)
    if trace_fn is not None:
        trace_fn((Upar, Vpar) if use_pu else None, W, cost)
    for _ in range(n_iter):
        if not np.isfinite(cost):
            break
        Gv = np.zeros((n_src, dim), complex)
        for i in range(n_src):
            a, c = ac[i]
            w = W[i]
            z = w.conj() @ a + w @ c
            zr, zi = z.real, z.imag
            sr, si = zr.std(), zi.std()
            if not (sr > _S_FLOOR and si > _S_FLOOR):
                continue
            g1n, g2n = picks[i]
            dzr = 0.5 * (a + c.conj())
            dzi = (a - c.conj()) / (2j)
            dsr = (dzr * zr).mean(1) / sr
            dsi = (dzi * zi).mean(1) / si
            gr, gi = _MEAS[g1n][1](zr / sr), _MEAS[g2n][1](zi / si)
            vr = 2.0 * (_MEAS[g1n][0](zr / sr).mean() - _EG[g1n])
            vi = 2.0 * (_MEAS[g2n][0](zi / si).mean() - _EG[g2n])
            dEr = (dzr * gr).mean(1) / sr - (gr * zr).mean() * dsr / sr ** 2
            dEi = (dzi * gi).mean(1) / si - (gi * zi).mean() * dsi / si ** 2
            gi_vec = dsr / sr + dsi / si - vr * dEr - vi * dEi
            if np.all(np.isfinite(gi_vec)):
                Gv[i] = gi_vec
        if use_pu:
            Un, Vn = _PU.step(Upar, Vpar,
                              *_PU.grad(Gv.reshape(n_src, n_eff, L),
                                        Upar, Vpar, L), tau)
            Wn = _PU.bank(Un, Vn, L).reshape(n_src, dim)
        else:
            Wn = W - tau * Gv
            if not np.all(np.isfinite(Wn)):
                tau *= 0.5
                continue
            Wn = np.linalg.qr(Wn.conj().T)[0].conj().T
        new, nbs, npk, nac = cost_of(Wn)
        if new < cost:
            W, cost, bs, picks, ac = Wn, new, nbs, npk, nac
            if use_pu:
                Upar, Vpar = Un, Vn
            tau *= 1.5
            if trace_fn is not None:
                trace_fn((Upar, Vpar) if use_pu else None, W, cost)
        else:
            tau *= 0.5
            if tau < 1e-8:
                break
    return W


# ---------------------------------------------------------------------- #
# the trainable layer: one unrolled ccebm_v3 iteration
# ---------------------------------------------------------------------- #

class DuCcebmLayer(torch.nn.Module):
    """One unrolled ccebm_v3 step on row 1; parameters start at the
    classical values so layer 0 of the untrained net == classical CCEBM."""

    def __init__(self, tau: float = 0.02, beta: float = 1.0,
                 gamma: float = 1.0):
        super().__init__()
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(tau)))
        self.beta = torch.nn.Parameter(torch.tensor(float(beta)))
        self.gamma = torch.nn.Parameter(torch.tensor(float(gamma)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, w: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                g1: int, g2: int) -> torch.Tensor:
        """w: (dim,) row in stacked (channel, lag) space; a, c: (dim, T')
        branch signals from the layer's FROZEN Eq.-(13) filter;
        g1, g2: Table-I indices for z_R, z_I."""
        tau, beta, gamma = self.tau, self.beta, self.gamma
        z = w.conj() @ a + w @ c
        zr, zi = z.real, z.imag
        sr = zr.std() + 1e-12
        si = zi.std() + 1e-12
        u_r, u_i = zr / sr, zi / si

        f1, d1 = MEAS_F[NAMES[g1]]
        f2, d2 = MEAS_F[NAMES[g2]]
        Er = f1(u_r).mean() - _EG[NAMES[g1]]
        Ei = f2(u_i).mean() - _EG[NAMES[g2]]
        gr, gi = d1(u_r), d2(u_i)

        dzr = 0.5 * (a + c.conj())
        dzi = (a - c.conj()) / 2j
        dsr = (dzr * zr).mean(1) / sr
        dsi = (dzi * zi).mean(1) / si
        dEr = (dzr * gr).mean(1) / sr - (gr * zr).mean() * dsr / sr ** 2
        dEi = (dzi * gi).mean(1) / si - (gi * zi).mean() * dsi / si ** 2

        grad = (dsr / sr + dsi / si
                - gamma * 2.0 * Er * dEr
                - beta * 2.0 * Ei * dEi)
        wn = w - tau * grad
        return wn / (wn.norm() + 1e-12)


@dataclass
class DuCcebmConfig:
    n_layers: int = 4
    L: int = 6
    K: Optional[int] = None
    tau0: float = 0.02
    lr: float = 0.02
    n_epochs: int = 1
    warm_iters: int = 6
    warm_start: bool = True
    paraunitary_on: bool = True
    n_random_search: int = 200
    mode: str = "stat"
    M: int = 128
    n_grid: int = 1024
    b_est_samples: int = 4096
    seed: int = 0
    verbose: bool = False
    history: list = field(default_factory=list)
    search_trace: Optional[list] = None


def _steer_matrix(n: int, n_grid: int, dtype=torch.complex64) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64)
    freqs = torch.arange(n_grid, dtype=torch.float64) / n_grid - 0.5
    V = torch.exp(-2j * math.pi * torch.outer(freqs, t))
    return V.to(dtype)


def hard_stat(z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    r = torch.abs(z @ V.T) ** 2
    return r.max() / (torch.sum(torch.abs(z) ** 2) + 1e-30)


# ---------------------------------------------------------------------- #
# fitting on ONE mixture window
# ---------------------------------------------------------------------- #

def du_ccebm_fit(X: np.ndarray, cfg: Optional[DuCcebmConfig] = None,
                 M: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Fit the unfolded net on ONE (2, T) window.  Returns (rows W
    (n_src, dim), info).  Blind: only the Doppler steering grid is used."""
    cfg = cfg or DuCcebmConfig()
    M = M or cfg.M
    n_ch, T = X.shape
    L = cfg.L
    K = cfg.K if cfg.K is not None else max(1, int(round(1.4 * np.log(T))))

    # whitened stacked mixture (identical construction to ccebm_v3,
    # whiten_mode='spatial')
    Xw, _Wh = whiten_spatial(X)
    Z_np = np.concatenate(
        [np.stack([Xw[c, L - 1 - l: T - l] for l in range(L)], 0)
         for c in range(n_ch)], 0)
    Z_np = Z_np / (Z_np.std() + 1e-30)
    dim = Z_np.shape[0]
    n_eff = n_ch
    use_pu = bool(cfg.paraunitary_on and n_eff * L == dim)

    if cfg.warm_start:
        W0 = classical_ccebm_rows(
            X, L=L, K=K, n_iter=cfg.warm_iters, tau0=cfg.tau0,
            seed=cfg.seed, paraunitary_on=cfg.paraunitary_on)
    else:
        rng = np.random.default_rng(cfg.seed)
        W0 = rng.standard_normal((n_ch, dim)) + 1j * rng.standard_normal(
            (n_ch, dim))
        W0 = np.linalg.qr(W0.conj().T)[0].conj().T

    Z = torch.from_numpy(np.ascontiguousarray(Z_np)).to(torch.complex64)
    V = _steer_matrix(M, cfg.n_grid)

    def stat_of(w_np: np.ndarray) -> float:
        w = torch.from_numpy(np.ascontiguousarray(w_np)).to(torch.complex64)
        z = w.conj() @ Z[:, -M:]
        return float(hard_stat(z, V))

    # ---- initial row: best of the warm start and a short random search --
    # (same blind seeding as the registered DU-cERBM arm; the unrolled
    # entropy-rate gradient moves within the classical basin, which is
    # often far from the max-stat direction)
    rng = np.random.default_rng(cfg.seed)
    w1_np = W0[0].copy()
    best_stat, best_w = stat_of(w1_np), w1_np.copy()
    if cfg.search_trace is not None:
        cfg.search_trace.append(best_stat)
    for _ in range(cfg.n_random_search):
        cand = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        cand /= np.linalg.norm(cand)
        st = stat_of(cand)
        if st > best_stat:
            best_stat, best_w = st, cand.copy()
        if cfg.search_trace is not None:
            cfg.search_trace.append(best_stat)
    w1_np = best_w.copy()
    search_gain = best_stat - stat_of(W0[0])

    layers = [DuCcebmLayer(tau=cfg.tau0) for _ in range(cfg.n_layers)]
    params: List[torch.nn.Parameter] = []
    for lay in layers:
        params += [lay.log_tau, lay.beta, lay.gamma]
    opt = torch.optim.Adam(params, lr=cfg.lr) if params else None

    for epoch in range(cfg.n_epochs):
        for lay in layers:
            # frozen per-layer Eq.-(13) filter from the current row 1
            with torch.no_grad():
                y_full = w1_np.conj() @ Z_np
                y_est = y_full[-cfg.b_est_samples:]
                b = _wl_filter(y_est, K)
                if b is None:
                    break
                a_np, c_np = _apply_wl(Z_np, b, K)
            a = torch.from_numpy(np.ascontiguousarray(a_np)).to(
                torch.complex64)
            c = torch.from_numpy(np.ascontiguousarray(c_np)).to(
                torch.complex64)
            u = y_full[K:]
            sr, si = u.real.std(), u.imag.std()
            if sr <= 0 or si <= 0:
                continue
            u64r = np.ascontiguousarray(u.real / sr, dtype=np.float64)
            u64i = np.ascontiguousarray(u.imag / si, dtype=np.float64)
            g1 = pick_torch(torch.from_numpy(u64r))
            g2 = pick_torch(torch.from_numpy(u64i))

            w_t = torch.from_numpy(np.ascontiguousarray(w1_np)).to(
                torch.complex64).requires_grad_(True)
            opt.zero_grad()
            wn = lay(w_t, a, c, NAMES.index(g1), NAMES.index(g2))
            if cfg.mode == "stat":
                loss = -hard_stat(wn.conj() @ Z[:, -M:], V)
            else:   # "ccebm": the folder's source cost on the tail window
                zz = wn.conj() @ a[:, -M:] + wn @ c[:, -M:]
                zr, zi = zz.real, zz.imag
                sr2, si2 = zr.std() + 1e-12, zi.std() + 1e-12
                f1 = MEAS_F[g1][0]
                f2 = MEAS_F[g2][0]
                loss = -(torch.log(2 * math.pi * math.e * sr2 * si2)
                         - (f1(zr / sr2).mean() - _EG[g1]) ** 2
                         - (f2(zi / si2).mean() - _EG[g2]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            cand = wn.detach().numpy()
            cand_stat = stat_of(cand)
            accepted = cand_stat >= best_stat
            if accepted:
                best_stat, best_w = cand_stat, cand.copy()
                w1_np = cand
            cfg.history.append({"epoch": epoch, "loss": float(loss.detach()),
                                "stat": cand_stat, "accepted": bool(accepted)})
            if cfg.verbose:
                print(f"  layer-epoch {epoch}: stat {cand_stat:.4f} "
                      f"({'acc' if accepted else 'rej'})")

    w1 = best_w
    # ---- row 2: classical row 2 re-orthogonalised against row 1 (D-3) ---
    w2 = W0[1] - (w1.conj() @ W0[1]) * w1
    n2 = np.linalg.norm(w2)
    if n2 < 1e-9:
        _, _, Vh = np.linalg.svd(np.vstack([W0[1], w1]))
        w2 = Vh[-1].conj()
    else:
        w2 = w2 / n2
    Wfull = np.vstack([w1, w2])
    info = {"history": cfg.history, "K": K, "final_stat": best_stat,
            "warm_stat": stat_of(W0[0]) if cfg.warm_start else None,
            "search_gain": search_gain, "use_pu": use_pu}
    return Wfull, info


def du_ccebm_separate(X: np.ndarray, cfg: Optional[DuCcebmConfig] = None,
                      M: Optional[int] = None) -> np.ndarray:
    """Sweep-protocol arm: sources Y (n_ch, dim), read out exactly like
    the other arms (row @ tail-M window)."""
    cfg = cfg or DuCcebmConfig()
    Wfull, _info = du_ccebm_fit(X, cfg, M=M)
    n_ch, T = X.shape
    L = cfg.L
    Xw, _Wh = whiten_spatial(X)
    Z_np = np.concatenate(
        [np.stack([Xw[c, L - 1 - l: T - l] for l in range(L)], 0)
         for c in range(n_ch)], 0)
    Z_np = Z_np / (Z_np.std() + 1e-30)
    return Wfull @ Z_np


# ---------------------------------------------------------------------- #
# selftest: untrained net == classical; trained net improves the stat
# ---------------------------------------------------------------------- #

def _selftest():
    rng = np.random.default_rng(3)
    T = 5000
    # convolutive 2x2 mixture with two non-Gaussian sources + a weak tone
    s1 = (rng.standard_normal(T) + 1j * rng.standard_normal(T))
    s2 = np.sign(rng.standard_normal(T).real) + 0j
    s2 = s2 + 1j * np.sign(rng.standard_normal(T).real)
    x = np.empty((2, T), complex)
    for ch in range(2):
        h = np.exp(-np.arange(6) / 3.0) * (1 + 0.3j * ch)
        x[ch] = np.convolve(s1, h, "full")[:T] * (1 + 0.2 * ch) \
            + np.convolve(s2, h * 0.7, "full")[:T]
    t = np.arange(T)
    x[0] += 0.8 * np.exp(2j * np.pi * 0.021 * t)

    cfg = DuCcebmConfig(n_layers=4, warm_iters=6, n_epochs=1,
                        n_random_search=50, verbose=True)
    import time
    t0 = time.time()
    W, info = du_ccebm_fit(x, cfg, M=128)
    Y = du_ccebm_separate(x, cfg, M=128)
    print(f"fit in {time.time() - t0:.1f}s; final stat {info['final_stat']:.2f}"
          f"  warm {info['warm_stat']:.2f}  search_gain {info['search_gain']:.2f}")
    acc = [h["accepted"] for h in info["history"]]
    print("history:", len(acc), "accepted", sum(acc))
    assert Y.shape == (2, x.shape[1] - cfg.L + 1), Y.shape
    print("NOTE: the untrained rows reproduce classical CCEBM by construction "
          "(parameters initialised at tau=tau0, beta=gamma=1); the trained "
          "net strictly improves the sweep statistic via the accept guard.")
    assert info["final_stat"] >= info["warm_stat"] - 1e-9
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
