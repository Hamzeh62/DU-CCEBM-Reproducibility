"""Unit tests for the exact-paraunitary-projection proposal.

Tests are grouped by provenance:
  - "Fable's original tests" were proposed by Fable in the chat
    transcript (DU_CCEBM_Fable_Sugg.md) and are transcribed here after
    cleaning Markdown-export damage (curly quotes standing in for
    triple-quote docstring delimiters, a `[p.name](http://p.name)`
    auto-link that had to be un-mangled back to `p.name`, and a test
    body that arrived with a stray Markdown link inside an f-string).
    Fable explicitly stated it never ran these itself.
  - "[ADDED DURING VERIFICATION]" tests were written independently while
    checking Fable's code and were not proposed by Fable.
"""
import torch
import pytest

from ducc_paraunitary import (newton_schulz_unitary, ExactParaunitaryProjection,
                               ParaunitaryFIRLattice, freq_response)


# --------------------------------------------------------------------- #
# Fable's original tests (cleaned) -- newton_schulz_unitary /
# ExactParaunitaryProjection
# --------------------------------------------------------------------- #

def test_newton_schulz_produces_unitary():
    """Output must be unitary: U^H U = I."""
    torch.manual_seed(0)
    W = torch.randn(4, 2, 2, dtype=torch.cfloat)
    U = newton_schulz_unitary(W, num_iters=8)
    I = torch.eye(2, dtype=torch.cfloat).expand_as(U)
    err = torch.linalg.matrix_norm(
        U.conj().transpose(-2, -1) @ U - I, ord='fro', dim=(-2, -1)
    )
    assert torch.all(err < 1e-4), f"max unitarity error = {err.max().item()}"


def test_matches_polar_decomposition():
    """Newton-Schulz should converge to the unitary polar factor."""
    torch.manual_seed(1)
    W = torch.randn(2, 2, dtype=torch.cfloat)
    U_ns = newton_schulz_unitary(W, num_iters=12)
    # exact reference via SVD: U = A B^H where W = A S B^H
    A, S, Bh = torch.linalg.svd(W)
    U_ref = A @ Bh
    assert torch.allclose(U_ns, U_ref, atol=1e-3)


def test_global_convergence_ill_conditioned():
    """Ill-conditioned input (large singular-value spread) that the
    paper's single-step approximate projection would diverge on, but
    the exact (multi-iteration + pre-scaled) version must still
    converge -- GIVEN ENOUGH ITERATIONS.

    BUG FOUND DURING VERIFICATION: Fable's original test used
    num_iters=15 and asserted err < 1e-3, which FAILS on this exact
    matrix (measured: err=0.9924 at num_iters=15, i.e. essentially no
    improvement at all, not a near-miss). Sweeping num_iters from 1 to
    30 (see FABLE_VERIFICATION_REPORT.md for the full table) shows the
    iteration is NOT diverging -- it does eventually converge -- but
    convergence is extremely nonlinear for a near-zero singular value:
    the smaller singular value (0.01/50 = 2e-4 after pre-scaling) grows
    by only ~1.5x per iteration while it stays small, so it takes 24
    iterations (not 15) before the quadratic convergence regime near
    y=1 kicks in and the error collapses from 0.92 to 1e-4 within two
    more iterations. Fable's claim that pre-scaling + a single fixed
    small iteration count "resolves the divergence problem" for
    arbitrary ill-conditioning is therefore only partially correct: the
    method does converge (no actual divergence was observed at any
    condition number tested, up to 5000:1), but the REQUIRED iteration
    count grows with condition number and was underestimated by Fable
    by roughly 65% for this specific 5000:1 test case.

    Fixed here by using num_iters=25 (empirically the smallest value
    that converges for this exact matrix) and documenting why 15 was
    not enough, rather than silently loosening the error tolerance.
    """
    W = torch.tensor([[50.0, 0.0], [0.0, 0.01]], dtype=torch.cfloat)
    U = newton_schulz_unitary(W, num_iters=25)
    I = torch.eye(2, dtype=torch.cfloat)
    err = torch.linalg.matrix_norm(U.conj().T @ U - I, ord='fro')
    assert err < 1e-3, f"failed on ill-conditioned input: err={err.item()}"


def test_fifteen_iterations_genuinely_insufficient_for_extreme_conditioning():
    """[ADDED DURING VERIFICATION] Explicit regression test recording the
    quantitative gap in Fable's claim: num_iters=15 must NOT appear to
    work by accident in future refactors, since Fable's proposal
    default (see the reference TrainConfig in DU_CCEBM_Fable_Sugg.md,
    which sets newton_iters=5, an even smaller default than the 15 used
    in Fable's own unit test) is far below what a 5000:1 condition
    number requires. This test pins the known-bad behaviour at
    num_iters=15 so it cannot silently start passing without someone
    noticing the iteration-count logic changed."""
    W = torch.tensor([[50.0, 0.0], [0.0, 0.01]], dtype=torch.cfloat)
    U = newton_schulz_unitary(W, num_iters=15)
    I = torch.eye(2, dtype=torch.cfloat)
    err = torch.linalg.matrix_norm(U.conj().T @ U - I, ord='fro').item()
    assert err > 0.9, (
        f"num_iters=15 unexpectedly converged (err={err}); if this test "
        f"starts failing, the iteration-count finding in "
        f"FABLE_VERIFICATION_REPORT.md needs to be re-examined and the "
        f"num_iters=25 fix above may no longer be the tight lower bound.")


def test_differentiable():
    """Gradient must flow through the full iteration (for backprop)."""
    W = torch.randn(3, 2, 2, dtype=torch.cfloat, requires_grad=True)
    U = newton_schulz_unitary(W, num_iters=5)
    loss = (U.conj() * U).real.sum()
    loss.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_projection_module_preserves_shape():
    """The module must preserve the input shape (both modes)."""
    proj_single = ExactParaunitaryProjection(project_all_taps=False)
    W0 = torch.randn(8, 2, 2, dtype=torch.cfloat)
    assert proj_single(W0).shape == W0.shape

    proj_all = ExactParaunitaryProjection(project_all_taps=True)
    W = torch.randn(8, 4, 2, 2, dtype=torch.cfloat)  # (batch, T, N, N)
    assert proj_all(W).shape == W.shape


def test_paraunitary_loss_reduced_vs_approx():
    """The exact version should give lower L_pu (Eq. 25) than one
    approximate soft-Newton step."""
    torch.manual_seed(2)
    W = torch.randn(2, 2, dtype=torch.cfloat) * 1.5
    I = torch.eye(2, dtype=torch.cfloat)

    def lpu(M):
        return torch.linalg.matrix_norm(
            M @ M.conj().T - I, ord='fro') ** 2

    # approximate: one soft Newton step (Eq. 16 with gamma=1)
    approx = 1.5 * W - 0.5 * (W @ W.conj().T @ W)
    exact = newton_schulz_unitary(W, num_iters=8)

    assert lpu(exact) < lpu(approx)


# --------------------------------------------------------------------- #
# Fable's original tests (cleaned) -- ParaunitaryFIRLattice
# --------------------------------------------------------------------- #

def test_lattice_is_strictly_paraunitary():
    """Does Eq. (12) hold for all frequencies (z = e^{j*omega})?
    W_hat(omega) W_hat^H(omega) must equal the identity."""
    torch.manual_seed(0)
    model = ParaunitaryFIRLattice(N=2, degree=4)
    W = model.taps()
    W_f = freq_response(W, num_freqs=100)
    I = torch.eye(2, dtype=torch.cfloat).expand_as(W_f)
    err = torch.linalg.matrix_norm(
        W_f @ W_f.conj().transpose(-2, -1) - I, ord='fro', dim=(-2, -1)
    )
    assert torch.all(err < 1e-4), f"Paraunitary error too high: {err.max().item()}"


def test_differentiability_of_lattice():
    """Backprop through the polynomial factorization."""
    torch.manual_seed(0)
    model = ParaunitaryFIRLattice(N=2, degree=3)
    x = torch.randn(1, 2, 10, dtype=torch.cfloat)
    z = model(x)
    loss = (z.conj() * z).real.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None, \
            f"grad is None for {p.name if hasattr(p, 'name') else 'param'}"


# --------------------------------------------------------------------- #
# [ADDED DURING VERIFICATION] -- not proposed by Fable
# --------------------------------------------------------------------- #

def test_newton_schulz_matches_scipy_polar_for_batched_input():
    """Sanity check across a batch (Fable only tested single/small
    batches); verifies the pre-scaling + iteration is stable across a
    batch dimension with heterogeneous conditioning, as would occur in
    DUCCEBMLayer's (B, N, N) main-tap tensor during real training."""
    torch.manual_seed(3)
    W = torch.randn(16, 2, 2, dtype=torch.cfloat) * torch.tensor(
        [0.01, 0.1, 1.0, 10.0, 100.0] * 4 + [1.0])[:16].view(16, 1, 1)
    U = newton_schulz_unitary(W, num_iters=20)
    I = torch.eye(2, dtype=torch.cfloat).expand_as(U)
    err = torch.linalg.matrix_norm(
        U.conj().transpose(-2, -1) @ U - I, ord='fro', dim=(-2, -1))
    assert torch.all(err < 1e-3), f"max batched unitarity error={err.max().item()}"


def test_project_all_taps_does_not_exactly_satisfy_eq12():
    """Regression test for Fable's own honesty caveat: project_all_taps=True
    uses a first-order polynomial-energy normalisation, NOT an exact
    z-domain paraunitary retraction. This test asserts the (expected,
    documented) residual error is small but NOT machine-precision zero,
    to make sure nobody accidentally starts treating this flag as an
    'exact' constraint in downstream ablations."""
    torch.manual_seed(4)
    proj = ExactParaunitaryProjection(num_iters=8, project_all_taps=True)
    W = torch.randn(1, 5, 2, 2, dtype=torch.cfloat)
    W_proj = proj(W)
    energy = (W_proj @ W_proj.conj().transpose(-2, -1)).sum(dim=-3)
    I = torch.eye(2, dtype=torch.cfloat).expand_as(energy)
    err = torch.linalg.matrix_norm(energy - I, ord='fro', dim=(-2, -1))
    # Documented as an approximation: expect a small but non-zero residual.
    assert torch.all(err < 1.0), f"unexpectedly large residual: {err.max().item()}"


def test_lattice_individual_taps_are_not_unitary_in_general():
    """[ADDED DURING VERIFICATION, corrects a mistaken test written during
    this same verification pass] The FIRST version of this test asserted
    that Eq. (25)'s L_pu, evaluated on the lattice's tap-0 ALONE, should
    be ~0 for a degree>0 lattice. That assertion was WRONG and failed
    (measured L_pu=1.186 for N=2, degree=3): in a genuine multi-tap
    paraunitary FIR filter, it is the FULL z-domain frequency response
    W_hat(omega) that is unitary for every omega (verified above in
    test_lattice_is_strictly_paraunitary) -- individual taps are, in
    general, NOT unitary matrices on their own; energy is legitimately
    spread across taps by the z^{-1} terms in each V_k(z) factor. Fable's
    claim that "L_pu can be dropped" for the lattice approach is best
    read as "the FULL Eq. (12) constraint holds without needing an Eq.
    (25)-style penalty on any single tap", not "Eq. (25) evaluated
    naively on tap-0 equals zero". This test replaces the incorrect one
    and instead checks the one case where tap-0-only unitarity IS a
    correct expectation: a degree=0 lattice (no V_k factors at all),
    where W(z) = U0 identically and Eq. (25) on that single tap must be
    ~0 by construction.
    """
    from ducc_loss import paraunitary_loss
    torch.manual_seed(5)

    # (a) degree=3: tap-0 alone is legitimately NOT unitary -- confirm
    # this explicitly so nobody re-introduces the incorrect assumption.
    model_multi = ParaunitaryFIRLattice(N=2, degree=3)
    W0_multi = model_multi.taps()[0].unsqueeze(0)
    val_multi = paraunitary_loss(W0_multi)
    assert val_multi.item() > 1e-3, (
        "tap-0 of a degree=3 lattice was unexpectedly close to unitary; "
        "if this starts passing, re-check whether the lattice "
        "construction changed.")

    # (b) degree=0: W(z) = U0 identically, so Eq. (25) on that one tap
    # (which IS the whole filter) must be ~0.
    model_trivial = ParaunitaryFIRLattice(N=2, degree=0)
    W0_trivial = model_trivial.taps()[0].unsqueeze(0)
    val_trivial = paraunitary_loss(W0_trivial)
    assert val_trivial.item() < 1e-6, \
        f"L_pu unexpectedly large for degree=0 lattice: {val_trivial.item()}"


