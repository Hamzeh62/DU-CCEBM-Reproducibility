"""Vectorized classical CCEBM used by the revision-v2 experiments.

The implementation preserves the operations and stopping rule of
``du_ccebm_simulation.ClassicalCCEBM`` while batching every window-level
operation.  CPU execution uses joblib with ``n_jobs=-1`` over vectorized
sub-batches; CUDA, when available, processes the full batch on GPU and falls
back to CPU on any device error.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from joblib import Parallel, delayed, effective_n_jobs


NU2_SATURATION_THRESH = 499.0


@dataclass
class BatchRun:
    scores: np.ndarray
    nu2: np.ndarray
    iterations: np.ndarray
    elapsed_s: float
    device: str
    n_jobs: int


def _as_complex_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(x), device=device, dtype=torch.complex128)


class VectorizedClassicalCCEBM:
    """Batched equivalent of the project's scalar 50-iteration CCEBM."""

    def __init__(
        self,
        N: int = 2,
        L: int = 8,
        K: int = 5,
        max_iter: int = 50,
        tol: float = 1e-5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.N = int(N)
        self.L = int(L)
        self.K = int(K)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = torch.device(device)

    @staticmethod
    def _same_convolution_basis(x: torch.Tensor, taps: int) -> torch.Tensor:
        """Return basis reproducing ``np.convolve(x, h, mode='same')``."""
        B, N, T = x.shape
        out = torch.zeros((B, N, taps, T), dtype=x.dtype, device=x.device)
        centre = (taps - 1) // 2
        for lag in range(taps):
            shift = centre - lag
            if shift >= 0:
                if shift < T:
                    out[:, :, lag, : T - shift] = x[:, :, shift:]
            else:
                offset = -shift
                if offset < T:
                    out[:, :, lag, offset:] = x[:, :, : T - offset]
        return out

    @staticmethod
    def _signal_same_convolution_basis(z: torch.Tensor, taps: int) -> torch.Tensor:
        return VectorizedClassicalCCEBM._same_convolution_basis(z[:, None, :], taps)[:, 0]

    @staticmethod
    def _roll_basis(x: torch.Tensor, taps: int) -> torch.Tensor:
        return torch.stack([torch.roll(x, -lag, dims=-1) for lag in range(taps)], dim=2)

    @staticmethod
    def _demix(W: torch.Tensor, x_basis: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnml,bmlt->bnt", W, x_basis)

    def _prewhiten(self, X: torch.Tensor) -> torch.Tensor:
        T = X.shape[-1]
        cov = torch.einsum("bnt,bmt->bnm", X, X.conj()) / T
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.clamp_min(1e-10)
        white = vals.rsqrt().unsqueeze(-1) * vecs.conj().transpose(-2, -1)
        return torch.einsum("bij,bjt->bit", white, X)

    def _whitening_filter(self, z: torch.Tensor) -> torch.Tensor:
        B, T = z.shape
        r = torch.stack(
            [(z * torch.roll(z, k, dims=-1).conj()).mean(-1) for k in range(self.K + 1)],
            dim=-1,
        )
        ij = torch.arange(self.K, device=z.device)
        toeplitz_idx = (ij[:, None] - ij[None, :]).abs()
        R = r[:, toeplitz_idx]
        eye = torch.eye(self.K, device=z.device, dtype=z.dtype).unsqueeze(0)
        rhs = -r[:, 1 : self.K + 1]
        try:
            b_ar = torch.linalg.solve(R + 1e-8 * eye, rhs.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Device kernels can occasionally reject a numerically marginal
            # batch.  The least-squares fallback is batched and never changes
            # already successful windows one by one.
            b_ar = torch.linalg.lstsq(R + 1e-8 * eye, rhs.unsqueeze(-1)).solution.squeeze(-1)
        b = torch.cat([torch.ones((B, 1), dtype=z.dtype, device=z.device), b_ar], dim=1)
        b = b / (b.abs().square().sum(-1, keepdim=True).sqrt() + 1e-12)
        basis = self._signal_same_convolution_basis(z, self.K + 1)
        return torch.einsum("bk,bkt->bt", b, basis)

    def _gradient(self, zn: torch.Tensor, x_roll: torch.Tensor) -> torch.Tensor:
        T = zn.shape[-1]
        sigma_r = zn.real.std(-1, correction=0).clamp_min(1e-8)
        sigma_i = zn.imag.std(-1, correction=0).clamp_min(1e-8)
        g_r = torch.tanh(zn.real / sigma_r[:, None])
        g_i = torch.tanh(zn.imag / sigma_i[:, None])
        scale = float(T * T)
        grad_r = -torch.einsum("bt,bmlt->bml", g_r, x_roll.real) / scale
        grad_i = -torch.einsum("bt,bmlt->bml", g_i, x_roll.imag) / scale
        grad_r = grad_r / (sigma_r[:, None, None] + 1e-8)
        grad_i = grad_i / (sigma_i[:, None, None] + 1e-8)
        return torch.complex(grad_r, grad_i)

    @staticmethod
    def _line_search(
        Z: torch.Tensor,
        grad: torch.Tensor,
        x_basis: torch.Tensor,
        source: int,
        active: torch.Tensor,
        max_tries: int = 10,
    ) -> torch.Tensor:
        """Vectorized Armijo proxy search, algebraically equal to scalar cost calls."""
        B = Z.shape[0]
        cost_before = Z.abs().square().mean(-1).sum(-1)
        source_energy = Z[:, source].abs().square().mean(-1)
        grad_output = torch.einsum("bml,bmlt->bt", grad, x_basis)

        tau = torch.full((B,), 0.1, dtype=Z.real.dtype, device=Z.device)
        selected = torch.zeros_like(tau)
        unresolved = active.clone()
        for _ in range(max_tries):
            candidate = Z[:, source] - tau[:, None] * grad_output
            cost_try = cost_before - source_energy + candidate.abs().square().mean(-1)
            accept = unresolved & torch.isfinite(cost_try) & (cost_try < cost_before)
            selected = torch.where(accept, tau, selected)
            unresolved = unresolved & ~accept
            tau = torch.where(unresolved, tau * 0.5, tau)
        return torch.where(unresolved, tau, selected)

    @staticmethod
    def _paraunitary_projection(W: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
        for _ in range(n_iter):
            W0 = W[:, :, :, 0]
            W_new = 1.5 * W0 - 0.5 * torch.bmm(torch.bmm(W0, W0.conj().transpose(1, 2)), W0)
            W = torch.cat([W_new.unsqueeze(-1), W[:, :, :, 1:]], dim=-1)
        return W

    def fit_separate_batch(self, X: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate a batch without any Python loop over windows."""
        X = _as_complex_tensor(X, self.device)
        if X.ndim != 3 or X.shape[1] != self.N:
            raise ValueError(f"Expected X with shape (B,{self.N},T), got {tuple(X.shape)}")
        B = X.shape[0]
        X_w = self._prewhiten(X)
        x_basis = self._same_convolution_basis(X_w, self.L)
        x_roll = self._roll_basis(X_w, self.L)

        W = torch.zeros((B, self.N, self.N, self.L), dtype=X.dtype, device=X.device)
        diag = torch.arange(self.N, device=X.device)
        W[:, diag, diag, 0] = 1.0
        active = torch.ones(B, dtype=torch.bool, device=X.device)
        iterations = torch.full((B,), self.max_iter, dtype=torch.int32, device=X.device)

        for iteration in range(self.max_iter):
            W_prev = W.clone()
            for source in range(self.N):
                Z = self._demix(W, x_basis)
                zn_w = self._whitening_filter(Z[:, source])
                grad = self._gradient(zn_w, x_roll)
                tau = self._line_search(Z, grad, x_basis, source, active)
                W_mid = W.clone()
                W_mid[:, source] = W_mid[:, source] - tau[:, None, None] * grad
                W_projected = self._paraunitary_projection(W_mid)
                W = torch.where(active[:, None, None, None], W_projected, W)

            diff = (W - W_prev).abs().reshape(B, -1).amax(-1)
            converged = active & (diff < self.tol)
            iterations = torch.where(
                converged,
                torch.full_like(iterations, iteration + 1),
                iterations,
            )
            active = active & ~converged
            if not bool(active.any()):
                break

        Z = self._demix(W, x_basis)
        return Z, W, iterations


def k_shape_nu_batch(z: torch.Tensor) -> torch.Tensor:
    """Vectorized form of ``k_distribution_mom``'s shape estimate."""
    amp = z.abs()
    m1 = amp.mean(-1)
    m2 = amp.square().mean(-1)
    ratio = m2 / (m1.square() + 1e-12)
    log_ratio = ratio.clamp_min(1e-300).log()
    lo = torch.full_like(m1, 0.05)
    hi = torch.full_like(m1, 500.0)
    log_const = float(np.log(4.0 / np.pi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        log_val = log_const + mid.log() + 2.0 * (torch.lgamma(mid) - torch.lgamma(mid + 0.5))
        undershoot = log_val < log_ratio
        hi = torch.where(undershoot, mid, hi)
        lo = torch.where(undershoot, lo, mid)
    nu = 0.5 * (lo + hi)
    return torch.where(m1 < 1e-12, torch.ones_like(nu), nu)


def _solve_numpy_batch(
    X: np.ndarray,
    ref_median: float | None,
    ref_mad: float | None,
    N: int,
    L: int,
    K: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    solver = VectorizedClassicalCCEBM(N=N, L=L, K=K, max_iter=max_iter, tol=tol, device="cpu")
    with torch.inference_mode():
        Z, _, iterations = solver.fit_separate_batch(X)
        nu2 = k_shape_nu_batch(Z[:, 1])
        if ref_median is None or ref_mad is None:
            scores = None
        else:
            scores = torch.where(
                nu2 >= NU2_SATURATION_THRESH,
                torch.zeros_like(nu2),
                (nu2 - float(ref_median)).abs() / float(ref_mad),
            )
        return (
            None if scores is None else scores.cpu().numpy(),
            nu2.cpu().numpy(),
            iterations.cpu().numpy(),
        )


def _split_nonempty(X: np.ndarray, parts: int) -> list[np.ndarray]:
    return [part for part in np.array_split(X, min(parts, len(X)), axis=0) if len(part)]


def run_batched_ccebm(
    X: np.ndarray,
    ref_median: float | None = None,
    ref_mad: float | None = None,
    *,
    N: int = 2,
    L: int = 8,
    K: int = 5,
    max_iter: int = 50,
    tol: float = 1e-5,
    prefer_cuda: bool = True,
) -> BatchRun:
    """Run on CUDA if present, otherwise all CPU cores via joblib."""
    X = np.asarray(X, dtype=np.complex128)
    start = time.perf_counter()

    if prefer_cuda and torch.cuda.is_available():
        try:
            scores, nu2, iterations = _solve_on_device(
                X, ref_median, ref_mad, N, L, K, max_iter, tol, torch.device("cuda")
            )
            return BatchRun(scores, nu2, iterations, time.perf_counter() - start, "cuda", 1)
        except (RuntimeError, MemoryError) as exc:
            print(f"[CCEBM] CUDA path failed ({exc}); falling back to CPU.", flush=True)
            torch.cuda.empty_cache()

    cpu_count = max(1, os.cpu_count() or 1)
    chunks = _split_nonempty(X, cpu_count)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        outputs = Parallel(n_jobs=-1, prefer="threads", require="sharedmem")(
            delayed(_solve_numpy_batch)(
                part, ref_median, ref_mad, N, L, K, max_iter, tol
            )
            for part in chunks
        )
    finally:
        torch.set_num_threads(previous_threads)

    score_parts = [item[0] for item in outputs]
    scores = None if score_parts[0] is None else np.concatenate(score_parts)
    nu2 = np.concatenate([item[1] for item in outputs])
    iterations = np.concatenate([item[2] for item in outputs])
    return BatchRun(
        scores=scores,
        nu2=nu2,
        iterations=iterations,
        elapsed_s=time.perf_counter() - start,
        device="cpu",
        n_jobs=effective_n_jobs(-1),
    )


def _solve_on_device(
    X: np.ndarray,
    ref_median: float | None,
    ref_mad: float | None,
    N: int,
    L: int,
    K: int,
    max_iter: int,
    tol: float,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    solver = VectorizedClassicalCCEBM(N=N, L=L, K=K, max_iter=max_iter, tol=tol, device=device)
    with torch.inference_mode():
        Z, _, iterations = solver.fit_separate_batch(X)
        nu2 = k_shape_nu_batch(Z[:, 1])
        scores = None
        if ref_median is not None and ref_mad is not None:
            scores_t = torch.where(
                nu2 >= NU2_SATURATION_THRESH,
                torch.zeros_like(nu2),
                (nu2 - float(ref_median)).abs() / float(ref_mad),
            )
            scores = scores_t.cpu().numpy()
        return scores, nu2.cpu().numpy(), iterations.cpu().numpy()


def calibrate_reference(
    clutter_X: np.ndarray,
    n_calib: int = 50,
    seed: int = 42,
    **solver_kwargs,
) -> tuple[float, float, dict]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(clutter_X), size=min(n_calib, len(clutter_X)), replace=False)
    run = run_batched_ccebm(clutter_X[chosen], **solver_kwargs)
    valid = run.nu2[run.nu2 < NU2_SATURATION_THRESH]
    if not len(valid):
        valid = np.asarray([1.0])
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)) * 1.4826 + 1e-6)
    audit = {
        "n_requested": int(n_calib),
        "n_used": int(len(chosen)),
        "indices": chosen.astype(int).tolist(),
        "n_saturated_excluded": int(np.sum(run.nu2 >= NU2_SATURATION_THRESH)),
        "elapsed_s": float(run.elapsed_s),
        "device": run.device,
        "n_jobs": int(run.n_jobs),
    }
    return median, mad, audit

