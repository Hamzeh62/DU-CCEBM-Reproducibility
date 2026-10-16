"""
cERBM -- complex entropy rate bound minimisation.

G.-S. Fu, R. Phlypo, M. Anderson and T. Adali, "Complex Independent
Component Analysis Using Three Types of Diversity: Non-Gaussianity,
Nonwhiteness, and Noncircularity", IEEE Trans. Signal Process. 63(3),
794-809, 2015.  Algorithm 1, with the closed-form whitening filter
(the cERBM-L variant the paper defines alongside it).

WHAT THE ALGORITHM EXPLOITS, AND HOW EACH PIECE APPEARS HERE
  non-Gaussianity   the four measuring functions of Table I, the
                    tightest bound chosen per variable
  sample dependence a per-source WIDELY LINEAR whitening filter
                    b = [p; q] in C^{2(K+1)}, z(t) = p^H y_(K+1)(t)
                    + q^H y*_(K+1)(t), closed form as the eigenvector of
                    R_aug^-1 D_b for the largest eigenvalue
  noncircularity    z_R and z_I bounded SEPARATELY, each with its own
                    measuring function, and the widely linear filter
                    itself

Cost, Eq. (2):  J_r(W) = sum_i H_r(y_i) - 2 log|det W|
Entropy rate bound, Sec. IV:
    H(z) <= log(2 pi e sigma_zR sigma_zI) - V1{G1(zbar_R)}
                                          - V2{G2(zbar_I)}

DECLARED DEVIATION
The paper's V(.) is the negentropy of the maximum-entropy density fitted
to each measuring function, which needs a per-call ME solve. This
implementation uses the standard quadratic negentropy approximation
V ~ (E{G(n)} - E{G(nu)})^2 against a Gaussian reference computed once by
Monte Carlo. The three diversities, the widely linear whitening filter,
the per-variable choice among the four measuring functions and the
decoupled gradient step of Eq. (16) are all as published; only the
entropy functional is approximated. Stated rather than buried.
"""
from __future__ import annotations

import numpy as np

# Table I of the paper
_MEAS = {
    "G1": (lambda u: u ** 4,
           lambda u: 4.0 * u ** 3),
    "G2": (lambda u: np.abs(u) / (1.0 + np.abs(u)),
           lambda u: np.sign(u) / (1.0 + np.abs(u)) ** 2),
    "G3": (lambda u: u * np.abs(u) / (10.0 + np.abs(u)),
           lambda u: np.abs(u) * (20.0 + np.abs(u))
           / (10.0 + np.abs(u)) ** 2),
    "G4": (lambda u: u / (1.0 + u ** 2),
           lambda u: (1.0 - u ** 2) / (1.0 + u ** 2) ** 2),
}
_NU = np.random.default_rng(0).standard_normal(400_000)
_EG = {k: float(v[0](_NU).mean()) for k, v in _MEAS.items()}


def _negentropy(u_std, name):
    """Quadratic approximation to V{E{G(.)}}; larger means a tighter
    bound, which is how the paper selects among the four functions."""
    return float((_MEAS[name][0](u_std).mean() - _EG[name]) ** 2)


def _pick(u_std):
    return max(_MEAS, key=lambda k: _negentropy(u_std, k))


def whiten(X, eps=1e-10):
    """Complex prewhitening: E{y y^H} = I."""
    mu = X.mean(1, keepdims=True)
    Xc = X - mu
    C = (Xc @ Xc.conj().T) / Xc.shape[1]
    d, E = np.linalg.eigh(C)
    keep = d > eps * d.max()
    Wh = (E[:, keep] / np.sqrt(d[keep])).conj().T
    return Wh @ Xc, Wh, mu


def _lagged(y, K):
    """y_(K+1)(t) = [y(t), y(t-1), ..., y(t-K)]^T, aligned to t >= K."""
    T = len(y)
    return np.stack([y[K - k: T - k] for k in range(K + 1)], 0)


def _closed_form_b(Y, K):
    """Widely linear whitening filter, closed form: the eigenvector of
    R_aug^-1 D_b for the largest eigenvalue (Sec. V, cERBM-L)."""
    Yl = _lagged(Y, K)                       # (K+1, T')
    Ya = np.vstack([Yl, Yl.conj()])          # augmented, 2(K+1)
    R = (Ya @ Ya.conj().T) / Ya.shape[1]
    R = R + 1e-8 * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    D = np.zeros(R.shape[0])
    D[0] = 1.0
    D[K + 1] = -1.0
    Mx = np.linalg.solve(R, np.diag(D))
    w, V = np.linalg.eig(Mx)
    b = V[:, int(np.argmax(w.real))]
    s = (b.conj() @ (D * b)).real            # enforce |p0|^2 - |q0|^2 = 1
    if abs(s) < 1e-12:
        b = np.zeros(R.shape[0], complex)
        b[0] = 1.0
    else:
        b = b / np.sqrt(abs(s)) * (1.0 if s > 0 else 1.0)
    return b, Ya


def _filter_mix(X, b, K):
    """a(t) and c(t) with z(t) = w^H a(t) + w^T c(t): the mixtures put
    through the widely linear whitening filter."""
    p, q = b[:K + 1], b[K + 1:]
    T = X.shape[1]
    a = np.zeros((X.shape[0], T - K), complex)
    c = np.zeros_like(a)
    for k in range(K + 1):
        a += np.conj(p[k]) * X[:, K - k: T - k]
        c += np.conj(q[k]) * np.conj(X[:, K - k: T - k])
    return a, c


def _perp(W, i):
    """Unit vector orthogonal to every row of W except row i (Eq. 16)."""
    other = np.delete(W, i, axis=0)
    if other.shape[0] == 0:
        h = W[i].conj()
    else:
        _, _, Vh = np.linalg.svd(other)
        h = Vh[-1].conj()
    n = np.linalg.norm(h)
    return h / n if n > 0 else h


def _source_cost(z):
    zr, zi = z.real, z.imag
    sr, si = zr.std(), zi.std()
    if sr <= 0 or si <= 0:
        return np.inf, None, None
    g1, g2 = _pick(zr / sr), _pick(zi / si)
    return (float(np.log(2 * np.pi * np.e * sr * si)
                  - _negentropy(zr / sr, g1) - _negentropy(zi / si, g2)),
            g1, g2)


def cerbm(X, K=None, n_iter=60, tau0=0.02, seed=0):
    """X: (N, T) complex mixtures. Returns (Y, info).

    Y are the separated sources in the whitened space, aligned to the
    last T-K samples of X.
    """
    N, T = X.shape
    if K is None:
        K = max(1, int(round(1.4 * np.log(T))))
    Z, Wh, mu = whiten(X)
    rng = np.random.default_rng(seed)
    W = np.linalg.qr(rng.standard_normal((N, N))
                     + 1j * rng.standard_normal((N, N)))[0]
    tau = tau0

    def total_cost(Wm):
        tot, picks, bs = 0.0, [], []
        for i in range(N):
            y = Wm[i].conj() @ Z
            b, _ = _closed_form_b(y, K)
            a, c = _filter_mix(Z, b, K)
            z = Wm[i].conj() @ a + Wm[i] @ c
            J, g1, g2 = _source_cost(z)
            tot += J
            picks.append((g1, g2))
            bs.append(b)
        sign, logdet = np.linalg.slogdet(Wm)
        return tot - 2.0 * logdet, picks, bs

    cost, picks, bs = total_cost(W)
    for _ in range(n_iter):
        Wn = W.copy()
        for i in range(N):
            b = bs[i]
            a, c = _filter_mix(Z, b, K)
            w = W[i]
            z = w.conj() @ a + w @ c
            zr, zi = z.real, z.imag
            sr, si = zr.std(), zi.std()
            if sr <= 0 or si <= 0:
                continue
            g1n, g2n = picks[i]
            # Wirtinger derivatives of z_R and z_I wrt w*
            dzr = 0.5 * (a + c.conj())
            dzi = (a - c.conj()) / (2j)
            dsr = (dzr * zr).mean(1) / sr
            dsi = (dzi * zi).mean(1) / si
            gr = _MEAS[g1n][1](zr / sr)
            gi = _MEAS[g2n][1](zi / si)
            vr = 2.0 * (_MEAS[g1n][0](zr / sr).mean() - _EG[g1n])
            vi = 2.0 * (_MEAS[g2n][0](zi / si).mean() - _EG[g2n])
            dEr = (dzr * gr).mean(1) / sr - (gr * zr).mean() * dsr / sr ** 2
            dEi = (dzi * gi).mean(1) / si - (gi * zi).mean() * dsi / si ** 2
            h = _perp(W, i)
            denom = np.vdot(h, w)
            grad = (dsr / sr + dsi / si - vr * dEr - vi * dEi
                    - h / (denom if abs(denom) > 1e-12 else 1e-12))
            Wn[i] = w - tau * grad
            n = np.linalg.norm(Wn[i])
            if n > 0:
                Wn[i] = Wn[i] / n
        new_cost, new_picks, new_bs = total_cost(Wn)
        if new_cost < cost:
            change = N - abs(np.trace(W.conj() @ Wn.T))
            W, cost, picks, bs = Wn, new_cost, new_picks, new_bs
            tau *= 1.5
            if change < 1e-7 * N:
                break
        else:
            tau *= 0.5
            if tau < 1e-8:
                break
    Y = W.conj() @ Z[:, K:] if K > 0 else W.conj() @ Z
    return Y, {"K": K, "cost": cost, "tau": tau,
               "measuring_functions": picks}
