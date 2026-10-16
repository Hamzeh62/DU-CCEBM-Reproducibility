"""Unit tests for the ANMF baseline added in Stage 2
(FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md).

These are written and run in this session (not transcribed from anyone
else's proposal) since ANMF was implemented from scratch here, guided
only by the closed-form definition given in the implementation prompt.
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '.')
from du_ccebm_simulation import (estimate_polarimetric_covariance,
                                  anmf_detect, anmf_detect_batch)


def _make_clutter_pool(rng, n=300, N=2, T=64, rho=0.6):
    """Correlated complex Gaussian clutter pool with HH/VV correlation
    rho, matching the kind of polarimetric structure ANMF is meant to
    exploit."""
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]], dtype=complex))
    pool = []
    for _ in range(n):
        z = (rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T))) / np.sqrt(2)
        pool.append(L @ z)
    return pool


def test_covariance_recovers_known_correlation():
    """R_hat estimated from a pool with known HH/VV correlation rho
    should recover rho (off-diagonal / sqrt(diag product)) reasonably
    well as the pool size grows."""
    rng = np.random.default_rng(0)
    rho_true = 0.6
    pool = _make_clutter_pool(rng, n=500, N=2, T=64, rho=rho_true)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=500, diag_loading=1e-3)
    rho_est = R_hat[0, 1].real / np.sqrt(R_hat[0, 0].real * R_hat[1, 1].real)
    assert abs(rho_est - rho_true) < 0.1, f"rho_est={rho_est}, expected ~{rho_true}"

    # R_inv must actually be the matrix inverse of R_hat
    N = R_hat.shape[0]
    assert np.allclose(R_hat @ R_inv, np.eye(N), atol=1e-6)


def test_covariance_is_hermitian_positive_definite():
    rng = np.random.default_rng(1)
    pool = _make_clutter_pool(rng, n=200)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=200)
    assert np.allclose(R_hat, R_hat.conj().T, atol=1e-10), "R_hat not Hermitian"
    eigvals = np.linalg.eigvalsh(R_hat)
    assert np.all(eigvals > 0), f"R_hat not positive definite: eigvals={eigvals}"


def test_anmf_single_matches_batch():
    """anmf_detect (per-sample loop, ground truth) and anmf_detect_batch
    (vectorised, used for large-scale table generation) must agree
    exactly (up to floating point) on identical inputs."""
    rng = np.random.default_rng(2)
    pool = _make_clutter_pool(rng, n=100)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=100)

    X = np.stack(_make_clutter_pool(rng, n=20, T=32), axis=0)  # (20, 2, 32)
    scores_loop = np.array([anmf_detect(X[i], R_inv) for i in range(X.shape[0])])
    scores_batch = anmf_detect_batch(X, R_inv)
    assert np.allclose(scores_loop, scores_batch, atol=1e-10), \
        f"max diff = {np.max(np.abs(scores_loop - scores_batch))}"


def test_anmf_bounded_in_unit_interval():
    """Lambda_ANMF is, by the Cauchy-Schwarz-like structure of its
    definition, bounded in [0, 1] for every pulse (equality only when x
    is exactly proportional to R^{-1}p); the time-averaged score must
    therefore also lie in [0, 1]."""
    rng = np.random.default_rng(3)
    pool = _make_clutter_pool(rng, n=200)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=200)
    X = np.stack(_make_clutter_pool(rng, n=50, T=64), axis=0)
    scores = anmf_detect_batch(X, R_inv)
    assert np.all(scores >= -1e-9) and np.all(scores <= 1.0 + 1e-9), \
        f"scores out of [0,1]: min={scores.min()}, max={scores.max()}"


def test_anmf_separates_matched_steering_target_from_clutter():
    """Sanity check of detection capability: inject a coherent signal
    aligned with the steering vector p=[1,1]/sqrt(2) into otherwise
    clutter-only windows, at a healthy SCR, and confirm ANMF scores are
    systematically higher for target-bearing windows than clutter-only
    windows (this is the whole point of a matched-filter-family
    detector; if this failed, the implementation would be suspect no
    matter how clean the algebra looks)."""
    rng = np.random.default_rng(4)
    N, T = 2, 64
    pool = _make_clutter_pool(rng, n=400, N=N, T=T, rho=0.5)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=400)

    clutter_test = np.stack(_make_clutter_pool(rng, n=100, N=N, T=T, rho=0.5), axis=0)
    p = np.ones(N, dtype=complex) / np.sqrt(N)
    scr_amplitude = 3.0  # healthy SCR, well above clutter RMS
    target_test = clutter_test.copy()
    for i in range(target_test.shape[0]):
        phase = rng.uniform(0, 2 * np.pi)
        target_test[i] += scr_amplitude * np.exp(1j * phase) * p[:, None]

    scores_clutter = anmf_detect_batch(clutter_test, R_inv)
    scores_target = anmf_detect_batch(target_test, R_inv)
    assert scores_target.mean() > scores_clutter.mean(), (
        f"target mean score ({scores_target.mean():.4f}) not > "
        f"clutter mean score ({scores_clutter.mean():.4f})")

    # NOTE on the assertion below (corrected after first running this
    # test): a naive "target_mean > 3x clutter_mean" check FAILED on the
    # first run (measured target_mean=0.860, clutter_mean=0.501) even
    # though the separation is in fact excellent (near-zero
    # distributional overlap, see the percentile check below). The
    # reason a 3x ratio is the wrong criterion: Lambda_ANMF is bounded
    # in [0,1] (see test_anmf_bounded_in_unit_interval), and for pure
    # clutter with N=2 secondary-estimated whitening, the statistic is
    # exactly the well-known "squared-cosine-similarity of an isotropic
    # random direction with a fixed direction" -- which has expectation
    # 1/N = 0.5 under H0 for N=2 (a Beta(1, N-1) distribution, the same
    # family underlying the classical ANMF closed-form Pfa formula
    # Pfa=(1-t)^(N-1)). The measured clutter mean of 0.501 matches this
    # theoretical value almost exactly, which is itself a strong
    # correctness check on the implementation -- but it also means a
    # ratio-based threshold like "3x" is mathematically impossible to
    # satisfy for N=2 once the true value saturates near the [0,1]
    # ceiling, regardless of how good the detector is. The correct way
    # to check "unambiguous separation" for a CFAR-type bounded
    # statistic is a percentile/overlap check, not a ratio of means.
    p5_target = np.percentile(scores_target, 5)
    p95_clutter = np.percentile(scores_clutter, 95)
    assert p5_target > p95_clutter, (
        f"distributions overlap: target 5th pct ({p5_target:.4f}) <= "
        f"clutter 95th pct ({p95_clutter:.4f})")


def test_anmf_score_independent_of_p_scaling():
    """Lambda_ANMF must be invariant to rescaling p by any nonzero
    complex constant (it appears both in numerator, squared, and
    denominator, squared -- the scale must cancel exactly)."""
    rng = np.random.default_rng(5)
    pool = _make_clutter_pool(rng, n=100)
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=100)
    X = np.stack(_make_clutter_pool(rng, n=10, T=16), axis=0)

    p1 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    p2 = p1 * (2.5 * np.exp(1j * 0.7))  # arbitrary nonzero complex rescale

    s1 = anmf_detect_batch(X, R_inv, p=p1)
    s2 = anmf_detect_batch(X, R_inv, p=p2)
    assert np.allclose(s1, s2, atol=1e-9)


def test_covariance_diag_loading_prevents_singular_matrix_for_tiny_pool():
    """With diag_loading=0 and a pool smaller than N (here N=2, pool
    size 1), the raw sample covariance from a SINGLE (N,T) window is
    generically singular (rank <= min(N, effective-samples-after-mean)
    only if T < N, but more relevantly: without loading, ill-conditioned
    pools can still blow up R_inv). This test only checks that the
    DEFAULT diag_loading keeps inversion well-behaved even for a
    minimal, adversarially small pool -- not a claim about the
    zero-loading case, which is intentionally not the default."""
    rng = np.random.default_rng(6)
    pool = _make_clutter_pool(rng, n=3, N=2, T=8)  # deliberately tiny pool
    R_hat, R_inv = estimate_polarimetric_covariance(pool, n_calib=3, diag_loading=1e-2)
    assert np.all(np.isfinite(R_inv)), "R_inv contains non-finite values"
    cond = np.linalg.cond(R_hat)
    assert cond < 1e6, f"R_hat unexpectedly ill-conditioned: cond={cond}"
