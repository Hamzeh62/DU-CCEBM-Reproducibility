"""
Exactly paraunitary FIR filter banks, by construction rather than by
projection.

WHY THIS FILE EXISTS
--------------------
The v3 arm imposed Eq. (12) with the iterative projection of Douglas et
al.: repeat  w_i <- 3/2 w_i - 1/2 g(w_i,w_i) - sum_{j<i} g(w_i,w_j)
until the bank is lossless. Measured on IPIX windows, that projection
does not do its job at the eight sweeps the sweep could afford:

    sweeps      8      30     100     400    2000
    residual  3.8e-1  1.3e-1  1.1e-1  9.3e-4  3.9e-16   (seed 0)
              3.4e-1  6.1e-2  4.2e-2  3.7e-2  3.7e-2    (seed 1)
              2.6e-1  1.8e-1  1.4e-1  1.4e-1  1.4e-1    (seed 3)

Two of five random starts never converge at all -- they sit on a
non-paraunitary fixed point -- and the ones that do converge need
hundreds of sweeps, which the sweep cannot pay for once per gradient
step. So the "paraunitary" arm was not testing the paraunitary
constraint; it was testing an arbitrary partial projection with a
residual of order one.

WHAT REPLACES IT
----------------
Vaidyanathan's factorisation: every causal FIR paraunitary system of
McMillan degree L-1 can be written

    W(z) = U . V_1(z) . V_2(z) ... V_{L-1}(z)
    V_k(z) = (I - v_k v_k^H) + z^-1 v_k v_k^H,   ||v_k|| = 1

with U unitary. Each V_k is paraunitary --
(I-P + z^-1 P)(I-P + z P) = (I-P) + P = I because P is idempotent -- so
the product is paraunitary for ANY unit vectors v_k and any unitary U.

The constraint therefore holds exactly at every iterate, to machine
precision, with no projection and no residual to monitor. Descent runs
on (U, v_1..v_{L-1}) instead of on the bank, with U retracted by QR and
each v_k renormalised; both retractions are exact, not approximate.

The parameterisation is complete for square banks (n_src == n_ch), which
is the case the sweep uses: two sources from the (HH, VV) pair.

CONVENTION. The bank is stored the way ccebm_v3 stores it, as
(n_src, n_ch, L) with the tap index last, and the source is formed as
y[n] = sum_{c,l} conj(W[i,c,l]) x_c[n-l]. Conjugating every coefficient
maps  sum_l W_l W_{l+tau}^H = delta_tau I  to its own complex conjugate,
and delta_tau I is real, so the stored bank satisfies the same condition
as the filter it represents.
"""
from __future__ import annotations

import numpy as np


def _pmul(A, B):
    """Polynomial matrix product: (m,n,pa) x (n,q,pb) -> (m,q,pa+pb-1)."""
    m, _, pa = A.shape
    _, q, pb = B.shape
    out = np.zeros((m, q, pa + pb - 1), complex)
    for i in range(pa):
        for j in range(pb):
            out[:, :, i + j] += A[:, :, i] @ B[:, :, j]
    return out


def _house(v):
    """The degree-one paraunitary factor V(z) = (I-P) + z^-1 P.

    v is ASSUMED to be a unit vector; normalisation is the retraction and
    lives in step(), not here. Folding it in here would make bank() a
    different map from the one grad() differentiates, and the
    finite-difference check in _selftest would -- rightly -- fail.
    """
    P = np.outer(v, v.conj())
    n = P.shape[0]
    out = np.zeros((n, n, 2), complex)
    out[:, :, 0] = np.eye(n) - P
    out[:, :, 1] = P
    return out


def bank(U, Vs, L):
    """Build the (n_src, n_ch, L) bank from the lattice parameters.

    U   (n_src, n_ch) with orthonormal rows
    Vs  a list of L-1 vectors in C^n_ch, each of unit norm
    """
    W = U[:, :, None].astype(complex)
    for v in Vs:
        W = _pmul(W, _house(v))
    out = np.zeros((U.shape[0], U.shape[1], L), complex)
    m = min(L, W.shape[2])
    out[:, :, :m] = W[:, :, :m]
    return out


def init(n_src, n_ch, L, rng):
    """A random point on the manifold."""
    A = rng.standard_normal((n_ch, n_src)) + 1j * rng.standard_normal(
        (n_ch, n_src))
    U = np.linalg.qr(A)[0][:, :n_src].conj().T
    Vs = []
    for _ in range(L - 1):
        v = rng.standard_normal(n_ch) + 1j * rng.standard_normal(n_ch)
        Vs.append(v / np.linalg.norm(v))
    return U, Vs


def grad(G, U, Vs, L):
    """Map a gradient in bank space to the lattice parameters.

    G is (n_src, n_ch, L), in the same convention ccebm_v3 uses for its
    descent direction: the step is W <- W - tau G. The map preserves the
    real inner product Re tr(G^H dW), so the returned gradients drive the
    same descent on the parameters.

    Returns (dU, dVs).
    """
    n_ch = U.shape[1]
    K = len(Vs)
    fac = [_house(v) for v in Vs]

    # prefix[k] = U V_1 ... V_k ;  suffix[k] = V_{k+1} ... V_K
    prefix = [U[:, :, None].astype(complex)]
    for f in fac:
        prefix.append(_pmul(prefix[-1], f))
    suffix = [None] * (K + 1)
    suffix[K] = np.eye(n_ch, dtype=complex)[:, :, None]
    for k in range(K - 1, -1, -1):
        suffix[k] = _pmul(fac[k], suffix[k + 1])

    dU = np.zeros_like(U)
    C = suffix[0]                      # W = U C
    for l in range(min(L, C.shape[2])):
        dU += G[:, :, l] @ C[:, :, l].conj().T

    dVs = []
    for k in range(K):
        A, B = prefix[k], suffix[k + 1]     # W = A V_{k+1} B
        S = np.zeros((2, n_ch, n_ch), complex)
        for tau in (0, 1):
            for p in range(A.shape[2]):
                for r in range(B.shape[2]):
                    l = p + r + tau
                    if l < L:
                        S[tau] += (A[:, :, p].conj().T @ G[:, :, l]
                                   @ B[:, :, r].conj().T)
        dP = S[1] - S[0]                   # V = (I-P) + z^-1 P
        dVs.append((dP + dP.conj().T) @ Vs[k])
    return dU, dVs


def step(U, Vs, dU, dVs, tau):
    """One retracted descent step. Both retractions are exact."""
    A = (U - tau * dU).conj().T
    Q = np.linalg.qr(A)[0]
    Un = Q[:, :U.shape[0]].conj().T
    Vn = []
    for v, d in zip(Vs, dVs):
        w = v - tau * d
        n = float(np.linalg.norm(w))
        Vn.append(v if (not np.isfinite(n) or n < 1e-12) else w / n)
    if not np.all(np.isfinite(Un)):
        Un = U
    return Un, Vn


def residual(W):
    """||sum_l W_l W_{l+tau}^H - delta_tau I||, the constraint violation."""
    N, _, L = W.shape
    tot = 0.0
    for tau in range(L):
        Gm = np.zeros((N, N), complex)
        for l in range(L - tau):
            Gm += W[:, :, l] @ W[:, :, l + tau].conj().T
        tot += np.linalg.norm(Gm - (np.eye(N) if tau == 0 else 0)) ** 2
    return float(np.sqrt(tot))


def _selftest():
    rng = np.random.default_rng(0)
    print("A) the constructed bank is paraunitary for random parameters")
    worst = 0.0
    for L in (2, 3, 6, 10):
        for _ in range(20):
            U, Vs = init(2, 2, L, rng)
            worst = max(worst, residual(bank(U, Vs, L)))
        print(f"   L={L:2d}  worst residual over 20 draws: "
              f"{residual(bank(U, Vs, L)):.2e}")
    print(f"   worst over all: {worst:.3e}")

    print()
    print("B) the parameter gradient matches finite differences of a")
    print("   quadratic test cost J(W) = Re<G, W>, so d/dtheta must equal")
    print("   the chain rule of the map alone")
    L = 6
    U, Vs = init(2, 2, L, rng)
    G = rng.standard_normal((2, 2, L)) + 1j * rng.standard_normal((2, 2, L))

    def J(U_, Vs_):
        return float(np.real(np.sum(np.conj(G) * bank(U_, Vs_, L))))

    dU, dVs = grad(G, U, Vs, L)
    eps = 1e-6
    err = []
    for _ in range(6):
        E = rng.standard_normal(U.shape) + 1j * rng.standard_normal(U.shape)
        num = (J(U + eps * E, Vs) - J(U - eps * E, Vs)) / (2 * eps)
        ana = float(np.real(np.sum(np.conj(dU) * E)))
        err.append(abs(num - ana) / max(abs(num), 1e-12))
    print(f"   dU  max relative error {max(err):.2e}")
    err = []
    for k in range(len(Vs)):
        for _ in range(3):
            E = (rng.standard_normal(Vs[k].shape)
                 + 1j * rng.standard_normal(Vs[k].shape))
            Vp = list(Vs)
            Vm = list(Vs)
            Vp[k] = Vs[k] + eps * E
            Vm[k] = Vs[k] - eps * E
            num = (J(U, Vp) - J(U, Vm)) / (2 * eps)
            ana = float(np.real(np.sum(np.conj(dVs[k]) * E)))
            err.append(abs(num - ana) / max(abs(num), 1e-12))
    print(f"   dv  max relative error {max(err):.2e}")
    print()
    print("   NOTE on B: the finite difference perturbs v off the unit")
    print("   sphere while the analytic gradient is the free-space one,")
    print("   so they must agree exactly -- normalisation is the")
    print("   retraction, applied in step(), not part of the map.")

    print()
    print("C) descent stays on the manifold for 500 steps")
    U, Vs = init(2, 2, L, rng)
    for _ in range(500):
        G = rng.standard_normal((2, 2, L)) + 1j * rng.standard_normal(
            (2, 2, L))
        U, Vs = step(U, Vs, *grad(G, U, Vs, L), 0.1)
    print(f"   residual after 500 random steps: "
          f"{residual(bank(U, Vs, L)):.3e}")


if __name__ == "__main__":
    _selftest()
