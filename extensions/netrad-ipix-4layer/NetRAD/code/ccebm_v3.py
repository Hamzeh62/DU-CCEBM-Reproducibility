"""
CCEBM v3 -- the two deviations of the earlier arm repaired.

Classical CCEBM, Sec. III-F of the DU-CCEBM manuscript:
  Eq (8)   convolutive demixing W(z) = sum_tau W_tau z^-tau
  Eq (9)   cost J(W) = sum_n H_r(z_n) - 2 log|det W|
  Eq (10)  entropy rate via the entropy bound, measuring functions
           chosen adaptively, z_R and z_I bounded separately
  Eq (12)  PARAUNITARY constraint W(z) W^H(z^-1) = I, imposed with the
           iterative scheme of Douglas et al.
  Eq (13)  a per-source WIDELY LINEAR whitening filter b in C^{2(K+1)}

WHAT WAS WRONG BEFORE, AND WHY IT MATTERED
The v2 arm used

  * a REAL autoregressive filter on the real composite, not the widely
    linear complex filter of Eq (13); and
  * deflationary orthogonalisation, not the paraunitary constraint of
    Eq (12).

cERBM, added later, was implemented with the widely linear filter from
the start. So the two algorithms were not being compared on equal
footing: CCEBM was denied the very machinery its own paper specifies,
while cERBM had it. Any conclusion that cERBM beats CCEBM drawn from
that pairing is about the implementations, not the algorithms. v3
removes both asymmetries; the only remaining difference between the arms
is the one that actually distinguishes them -- CCEBM demixes
convolutively, cERBM instantaneously.

The same declared approximation as cERBM applies: the maximum-entropy
negentropy functional is replaced by the standard quadratic form.
"""
from __future__ import annotations

import numpy as np

import pu_lattice as PU
from cerbm import (_MEAS, _EG, _negentropy, _pick,
                   _lagged)


# Below this the 1/s and 1/s**2 in the gradient overflow float64.
_S_FLOOR = 1e-120


def _st_whiten(V, eps=1e-10):
    """Spatio-temporal prewhitening, Eq (4) of the STFICA reference."""
    mu = V.mean(1, keepdims=True)
    Vc = V - mu
    C = (Vc @ Vc.conj().T) / Vc.shape[1]
    d, E = np.linalg.eigh(C)
    keep = d > eps * d.max()
    return (E[:, keep] / np.sqrt(d[keep])).conj().T @ Vc


def _stack(X, L):
    """Stacked spatio-temporal vector, index (channel, lag) in C order."""
    n_ch, T = X.shape
    return np.concatenate(
        [np.stack([X[c, L - 1 - l: T - l] for l in range(L)], 0)
         for c in range(n_ch)], 0)


# The Douglas projection that used to live here is gone. Measured on
# IPIX windows it left a residual of 0.2 to 0.5 at the eight sweeps this
# sweep can afford, and two of five random starts never converged at
# all, so the arm was not imposing Eq. (12) -- see pu_lattice.py, which
# replaces it with Vaidyanathan's factorisation. That construction is
# paraunitary to 2e-15 for ANY parameter values, so there is nothing
# left to project and nothing left to diverge.


def paraunitary_residual(W, n_ch, L):
    """||sum_l W_l W_{l+tau}^H - delta_tau I||, the constraint violation."""
    N = W.shape[0]
    Wr = W.reshape(N, n_ch, L)
    tot = 0.0
    for tau in range(L):
        G = np.zeros((N, N), complex)
        for l in range(L - tau):
            G += Wr[:, :, l] @ Wr[:, :, l + tau].conj().T
        tot += np.linalg.norm(G - (np.eye(N) if tau == 0 else 0)) ** 2
    return float(np.sqrt(tot))


def _wl_filter(y, K):
    """Widely linear whitening filter, Eq (13), closed form -- the same
    machinery cERBM uses, which v2 denied to CCEBM."""
    if not np.all(np.isfinite(y)):
        return None
    Yl = _lagged(y, K)
    Ya = np.vstack([Yl, Yl.conj()])
    R = (Ya @ Ya.conj().T) / Ya.shape[1]
    if not np.all(np.isfinite(R)):
        return None
    R = R + 1e-8 * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    D = np.zeros(R.shape[0])
    D[0], D[K + 1] = 1.0, -1.0
    ev, V = np.linalg.eig(np.linalg.solve(R, np.diag(D)))
    b = V[:, int(np.argmax(ev.real))]
    s = (b.conj() @ (D * b)).real
    if abs(s) < 1e-12:
        b = np.zeros(R.shape[0], complex)
        b[0] = 1.0
    else:
        b = b / np.sqrt(abs(s))
    return b


def _apply_wl(Z, b, K):
    """z(t) = p^H y_(K+1)(t) + q^H y*_(K+1)(t) as a pair (a, c) with
    z = w^H a + w^T c, so the gradient stays explicit."""
    p, q = b[:K + 1], b[K + 1:]
    T = Z.shape[1]
    a = np.zeros((Z.shape[0], T - K), complex)
    c = np.zeros_like(a)
    for k in range(K + 1):
        a += np.conj(p[k]) * Z[:, K - k: T - k]
        c += np.conj(q[k]) * np.conj(Z[:, K - k: T - k])
    return a, c


def ccebm_v3(X, L=10, n_src=2, K=None, n_iter=30, tau0=0.02, seed=7,
             paraunitary_on=True, whiten_mode="st",
             n_restart=1):
    """X: (n_ch, T) complex. Returns (Y, info).

    whiten_mode "st"      spatio-temporal, Eq (4) of the STFICA reference
                "spatial" channels only, which is what the v2 arm used
    Section 14 of the NetRAD preregistration measured that spatio-temporal
    prewhitening leaves each extracted source already temporally white, so
    the entropy-rate filter of Eq (13) has nothing left to remove. With
    the widely linear filter now in place that matters much more than it
    did, so the choice is exposed rather than hard-coded.
    """
    # Multi-start. Under the exact paraunitary constraint the cost is
    # correlated with separation quality the right way round (-0.39 over
    # 20 seeds on a known convolutive mixture), so picking the lowest-cost
    # restart is a legitimate unsupervised selection: it never looks at
    # the target. Without the constraint the correlation is +0.50 -- the
    # wrong sign -- and selecting on cost would pick the WORST solution,
    # which is why restarts are only worth paying for on the pu arm.
    if n_restart > 1:
        best = None
        for r in range(n_restart):
            out = ccebm_v3(X, L=L, n_src=n_src, K=K, n_iter=n_iter,
                           tau0=tau0, seed=seed + 1009 * r,
                           paraunitary_on=paraunitary_on,
                           whiten_mode=whiten_mode, n_restart=1)
            if best is None or out[1]["cost"] < best[1]["cost"]:
                best = out
        best[1]["n_restart"] = n_restart
        return best

    n_ch, T = X.shape
    if K is None:
        K = max(1, int(round(1.4 * np.log(T))))
    if whiten_mode == "spatial":
        Xc = X - X.mean(1, keepdims=True)
        C = (Xc @ Xc.conj().T) / Xc.shape[1]
        d, E = np.linalg.eigh(C)
        keep = d > 1e-10 * d.max()
        Xw = (E[:, keep] / np.sqrt(d[keep])).conj().T @ Xc
        Z = _stack(Xw, L)
        Z = Z / (Z.std() + 1e-30)
        Z_n_ch = Xw.shape[0]
    else:
        Z = _st_whiten(_stack(X, L))
        Z_n_ch = 0
    dim = Z.shape[0]
    n_eff = Z_n_ch                      # channels that survived whitening
    rng = np.random.default_rng(seed)

    # The paraunitary constraint is a statement about a bank of filters
    # over CHANNELS and LAGS, so it only means anything while that
    # structure is still visible. Spatio-temporal whitening mixes
    # channels and lags together and can drop dimensions, which destroys
    # it; so does losing a channel to a rank-deficient spatial
    # covariance. Both cases fall back to the orthogonal constraint and
    # say so in the info dict rather than pretending.
    use_pu = bool(paraunitary_on and whiten_mode == "spatial"
                  and n_eff * L == dim and n_src <= n_eff)
    if use_pu:
        Upar, Vpar = PU.init(n_src, n_eff, L, rng)
        W = PU.bank(Upar, Vpar, L).reshape(n_src, dim)
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
            # A floor, not a zero test. The gradient carries 1/s and
            # 1/s**2, so a standard deviation of 1e-160 -- which a
            # widely linear filter that nearly annihilates its own
            # source can produce -- overflows to inf, and inf reaches
            # paraunitary normalisation as inf/inf = nan. That is how
            # two of the three v3 shards died.
            if not (sr > _S_FLOOR and si > _S_FLOOR):
                return np.inf, bs, picks, ac
            g1, g2 = _pick(zr / sr), _pick(zi / si)
            tot += (np.log(2 * np.pi * np.e * sr * si)
                    - _negentropy(zr / sr, g1) - _negentropy(zi / si, g2))
            bs.append(b)
            picks.append((g1, g2))
        if not np.isfinite(tot):
            return np.inf, bs, picks, ac
        return float(tot), bs, picks, ac

    cost, bs, picks, ac = cost_of(W)
    n_reject = 0
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
            Un, Vn = PU.step(Upar, Vpar,
                             *PU.grad(Gv.reshape(n_src, n_eff, L),
                                      Upar, Vpar, L), tau)
            Wn = PU.bank(Un, Vn, L).reshape(n_src, dim)
        else:
            Wn = W - tau * Gv
            if not np.all(np.isfinite(Wn)):
                tau *= 0.5
                n_reject += 1
                if tau < 1e-8:
                    break
                continue
            Wn = np.linalg.qr(Wn.conj().T)[0].conj().T
        new, nbs, npk, nac = cost_of(Wn)
        if new < cost:
            W, cost, bs, picks, ac = Wn, new, nbs, npk, nac
            if use_pu:
                Upar, Vpar = Un, Vn
            tau *= 1.5
        else:
            n_reject += 1
            tau *= 0.5
            if tau < 1e-8:
                break
    Wb = W.reshape(n_src, n_eff, L) if n_eff * L == dim else None
    return W.conj() @ Z, {
        "K": K, "cost": cost, "L": L,
        "paraunitary_on": bool(use_pu),
        "paraunitary_requested": bool(paraunitary_on),
        "n_rejected_steps": n_reject,
        "paraunitary_residual": (PU.residual(Wb) if Wb is not None
                                 else None),
        "measuring_functions": picks}
