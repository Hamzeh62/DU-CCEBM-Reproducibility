"""Exact / near-exact paraunitary projection utilities for DU-CCEBM.

Reference: du_ccebm_taes_TWOCOLUMN.pdf, Eqs. (12), (15)-(18), pp. 4-6.
Newton-Schulz iteration: N. J. Higham, "Functions of Matrices: Theory
and Computation," SIAM, 2008.
Lattice factorization: P. P. Vaidyanathan, "Multirate Systems and
Filter Banks," Prentice Hall, 1993, Ch. 14.

PROVENANCE NOTE (added during independent verification): this file is a
manually cleaned transcription of code that arrived embedded in a
Markdown chat transcript (DU_CCEBM_Fable_Sugg.md), produced by a
different model ("Fable") that explicitly stated it could not run or
test the code itself. Cleaning required, among other things: restoring
triple-double-quote docstring delimiters that the Markdown renderer had
replaced with typographic curly quotes in several places; reconstructing
test function bodies that had been split across separate paragraphs by
the transcript UI (plain Python comments turned into Markdown section
headers with '===' underlines, and inline strings using curly quotes
instead of straight ASCII quotes); un-mangling one Markdown auto-link,
back into the intended attribute access p.name.
No mathematical content was changed; see FABLE_VERIFICATION_REPORT.md
for the full account and pytest results.

Fable's own stated caveat (kept verbatim in spirit, not literally
translated): `project_all_taps=True` in ExactParaunitaryProjection is
Fable's OWN extrapolation beyond the paper -- the paper only ever
projects the single main tap W_0 (Eq. 16). The polynomial-energy
normalisation used there is a first-order approximation of the full
constraint sum_tau W_tau W_tau^H = I, not an exact retraction onto the
paraunitary manifold in the z-domain. ParaunitaryFIRLattice below is a
genuinely exact construction (verified in the tests), but it changes the
network's parametrisation (free lattice angles instead of a
per-layer W matrix + Wirtinger gradient step), so it is NOT a drop-in
replacement for the existing DUCCEBMLayer sub-op 4 without further
architectural surgery; it is verified here in isolation only, per Stage
0 of the implementation plan, and is NOT wired into du_ccebm_simulation.py
in this session (see FABLE_VERIFICATION_REPORT.md, "Scope of this
session" section).
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------------- #
# Newton-Schulz unitary projection (drop-in replacement candidate for
# DUCCEBMLayer's existing single-step approximate projection)
# --------------------------------------------------------------------- #

def newton_schulz_unitary(W: torch.Tensor,
                           num_iters: int = 5,
                           eps: float = 1e-7) -> torch.Tensor:
    r"""Exact, differentiable projection of a complex matrix onto the
    unitary manifold via Newton-Schulz iteration (no SVD).

    Converges to the unitary polar factor U in the polar decomposition
    W = U P, i.e. U = W (W^H W)^{-1/2}, the closest unitary matrix to W
    in Frobenius norm.

    Args:
        W:         complex tensor of shape (..., N, N).
        num_iters: number of Newton-Schulz iterations.
        eps:       numerical-stability constant for pre-scaling.

    Returns:
        Unitary matrix U with the same shape as W.

    Reference: [1] Higham, "Functions of Matrices," SIAM 2008.
    """
    # --- pre-scaling to guarantee global convergence ---
    # all singular values must lie in (0, sqrt(3));
    # dividing by the (over-)estimated spectral norm (Frobenius norm is
    # an upper bound on it) guarantees this.
    norm = torch.linalg.matrix_norm(W, ord='fro', dim=(-2, -1), keepdim=True)
    Y = W / (norm + eps)

    # --- Newton-Schulz iteration: Y_{k+1} = 3/2 Y - 1/2 Y (Y^H Y) ---
    for _ in range(num_iters):
        YhY = Y.conj().transpose(-2, -1) @ Y  # Y^H Y
        Y = 1.5 * Y - 0.5 * (Y @ YhY)

    return Y


class ExactParaunitaryProjection(nn.Module):
    r"""Differentiable replacement for the paper's approximate Eq. (16)
    projection.

    Instead of one soft Newton step on only the main tap W_0, this
    module:
      1. runs the FULL Newton-Schulz iteration (numerical stability),
      2. keeps the trainable blend parameter gamma to control projection
         strength (for continuity with the paper's formulation),
      3. optionally applies the projection to ALL taps W_tau, so the
         Eq. (12) constraint is respected more fully across the
         polynomial domain -- see the "Fable's own stated caveat"
         paragraph in the module docstring above regarding this option.
    """

    def __init__(self, num_iters: int = 5, project_all_taps: bool = False):
        super().__init__()
        self.num_iters = num_iters
        self.project_all_taps = project_all_taps
        # gamma_hat matches the paper's convention; gamma = sigmoid(gamma_hat) in (0,1)
        self.gamma_hat = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ~= 0.88

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W: complex filter tensor.
               If project_all_taps=False: shape (..., N, N), i.e. the
                                           main tap W_0.
               If project_all_taps=True:  shape (..., T, N, N), i.e. all
                                           taps.
        Returns:
            Projected filter, same shape as the input.
        """
        gamma = torch.sigmoid(self.gamma_hat)

        if self.project_all_taps:
            # Convolutional paraunitary constraint: push every tap toward
            # unitary, then renormalise total energy across the tap axis.
            U = newton_schulz_unitary(W, self.num_iters)
            W_proj = (1.0 - gamma) * W + gamma * U
            # Polynomial-energy normalisation: sum_tau W_tau W_tau^H ~= I
            energy = (W_proj @ W_proj.conj().transpose(-2, -1)).sum(dim=-3)
            N = W.shape[-1]
            scale = torch.sqrt(
                torch.tensor(N, dtype=energy.real.dtype, device=W.device)
                / (torch.linalg.matrix_norm(energy, ord='fro',
                                             dim=(-2, -1), keepdim=True) + 1e-7)
            ).unsqueeze(-3)
            return W_proj * scale
        else:
            # Consistent with the paper: only the main tap W_0, but with
            # the exact Newton iteration instead of a single soft step.
            U = newton_schulz_unitary(W, self.num_iters)
            return (1.0 - gamma) * W + gamma * U


# --------------------------------------------------------------------- #
# Polyphase / lattice factorization: paraunitary "by construction"
# --------------------------------------------------------------------- #

class ParaunitaryFIRLattice(nn.Module):
    r"""Exact paraunitary FIR filter bank via Vaidyanathan factorization:

        W(z) = V_D(z) ... V_1(z) U_0,
        V_k(z) = I - v_k v_k^H + z^{-1} v_k v_k^H,   ||v_k|| = 1.

    The paper's Eq. (12) constraint holds by construction; no projection
    (Eq. 16) or penalty (Eq. 25) is required.
    """

    def __init__(self, N: int = 2, degree: int = 3, num_ns_iters: int = 10):
        """
        num_ns_iters default CHANGED during independent verification from
        Fable's original value of 6 to 10. At num_ns_iters=6, the test
        `test_lattice_is_strictly_paraunitary` (torch.manual_seed(0))
        failed with a reproducible U0-unitarity residual of 2.1e-3 --
        the same class of "convergence-speed depends on the random
        matrix's conditioning, not just a fixed small iteration count"
        issue independently found and documented for
        `newton_schulz_unitary` itself in
        `test_global_convergence_ill_conditioned`. A sweep on the exact
        failing seed (see FABLE_VERIFICATION_REPORT.md) showed the
        residual collapses from 5.2e-2 at iter 5 to 2.1e-3 at iter 6 to
        <1e-7 by iter 9; num_ns_iters=10 leaves comfortable margin.
        """
        super().__init__()
        self.N = N
        self.degree = degree  # D = L-1; number of taps = degree+1
        self.num_ns_iters = num_ns_iters

        self.U0_free = nn.Parameter(
            torch.randn(N, N, dtype=torch.cfloat) / N ** 0.5
        )
        self.v_free = nn.Parameter(
            torch.randn(degree, N, dtype=torch.cfloat)
        )

    def taps(self) -> torch.Tensor:
        """Compute the tap tensor W_tau from the lattice parameters.

        Returns: complex tensor of shape (degree+1, N, N).
        """
        U0 = newton_schulz_unitary(self.U0_free, self.num_ns_iters)

        # start: W(z) = U0 (main tap only)
        W = U0.unsqueeze(0)  # (1, N, N)

        eps = 1e-7
        for k in range(self.degree):
            v = self.v_free[k]
            v = v / (torch.linalg.vector_norm(v) + eps)
            P = torch.outer(v, v.conj())          # v v^H, rank-1 projector
            Q = torch.eye(self.N, dtype=P.dtype, device=P.device) - P

            # polynomial multiplication: [Q + z^{-1} P] * W(z)
            # new tap t: Q @ W_t + P @ W_{t-1}
            zeros = torch.zeros_like(W[:1])
            W_pad_lo = torch.cat([W, zeros], dim=0)       # W_t,   t=0..T
            W_pad_hi = torch.cat([zeros, W], dim=0)       # W_{t-1}
            W = Q @ W_pad_lo + P @ W_pad_hi                # (T+1, N, N)

        return W  # (degree+1, N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional demixer: z(t) = sum_tau W_tau x(t - tau).

        Args:
            x: complex signal of shape (B, N, T) -- e.g. HH/VV channels.
        Returns:
            Separated signal of shape (B, N, T).
        """
        W = self.taps()                                   # (L, N, N)
        L = W.shape[0]
        # conv1d kernel: (out=N, in=N, L); taps reversed for convolution
        kernel = W.flip(0).permute(1, 2, 0)
        x_pad = torch.nn.functional.pad(x, (L - 1, 0))    # causal
        return torch.nn.functional.conv1d(x_pad, kernel)


def freq_response(W_taps: torch.Tensor, num_freqs: int = 64) -> torch.Tensor:
    """W_hat(omega_m) = sum_t W_t e^{-j*omega_m*t} for all frequencies.

    Args:
        W_taps: complex tensor (L, N, N).
        num_freqs: number of frequency samples over [0, 2*pi).
    Returns:
        complex tensor (num_freqs, N, N).
    """
    # BUG FOUND DURING VERIFICATION (present in Fable's original code
    # verbatim): the einsum pattern 'fl,lnn->fnn' is invalid -- 'n'
    # cannot appear twice in the *output* subscript of torch.einsum
    # without being summed/diagonal-reduced, and PyTorch raises
    # "output subscript n appears more than once in the output" at
    # runtime. Fixed by using distinct labels (n, m) for the two matrix
    # dimensions, which is what was actually intended (a full N x N
    # matrix per frequency, not its diagonal).
    L = W_taps.shape[0]
    t = torch.arange(L, dtype=torch.float32)
    w = torch.linspace(0, 2 * torch.pi, num_freqs)
    phase = torch.exp(-1j * torch.outer(w, t)).to(torch.complex64)  # (num_freqs, L)
    return torch.einsum('fl,lnm->fnm', phase, W_taps)
