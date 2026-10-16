"""Unit / sanity tests for the DU-CCEBM loss extensions.

Tests 1-6 below are Fable's original proposed tests, transcribed from the
Markdown transcript (DU_CCEBM_Fable_Sugg.md) and cleaned of export
artifacts, but NOT otherwise altered in intent. Fable stated explicitly
it could not run these itself. Tests marked "[ADDED DURING
VERIFICATION]" were written independently while checking Fable's code
and were NOT proposed by Fable; they specifically probe the two
weaknesses flagged in FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md
(the batch-size/alpha mismatch in partial_auc_loss and an unguarded
divide-by-zero in mmd_loss for degenerate batches).
"""
import torch
import pytest

from ducc_loss import (mmd_loss, partial_auc_loss, paraunitary_loss,
                        detection_loss, DUCCEBMLoss, k_hard_negatives)

torch.manual_seed(0)


# --------------------------------------------------------------------- #
# Fable's original tests (cleaned)
# --------------------------------------------------------------------- #

def test_mmd_zero_for_identical_distributions():
    """MMD^2 of a distribution with itself -> ~0 (up to sampling noise).

    BUG FOUND DURING VERIFICATION (in Fable's original test, not just an
    implementation detail): Fable's version called
    `mmd_loss(x, x.clone())`, i.e. compared a batch against an EXACT
    duplicate of itself (same values, not an independent draw). Because
    Fable's k_xy term is never diagonal-excluded (only k_xx/k_yy are,
    per the standard unbiased-MMD^2 formula), every diagonal entry of
    k_xy is then a *self*-kernel equal to 1 (the maximum possible
    value), which inflates mean(k_xy) above its true cross-population
    expectation and biases the whole estimator NEGATIVE. This is not
    sampling noise -- it is a small but highly reproducible bias
    (measured empirically: -0.0181 +/- 0.0002 across 30 seeds at
    n=256, i.e. a real, systematic effect, not variance), caused purely
    by the choice of test data (identical duplicate rows), and it
    exceeded Fable's own stated 1e-3 tolerance by ~18x.

    Fixed here by comparing two INDEPENDENT draws from the same
    distribution (the only correct way to test "MMD^2 of a distribution
    with itself"), at a larger n to reduce sampling variance (measured
    empirically: std ~= 7e-4 at n=1024 across 50 seeds), with a
    tolerance loose enough to comfortably cover that variance without
    being vacuous.
    """
    torch.manual_seed(0)
    x = torch.randn(1024, 8)
    y = torch.randn(1024, 8)   # independent draw, NOT x.clone()
    val = mmd_loss(x, y)
    assert abs(val.item()) < 5e-3


def test_mmd_positive_for_shifted_distributions():
    """MMD^2 grows when the two batches are clearly separated."""
    x = torch.randn(256, 8)
    y = torch.randn(256, 8) + 5.0
    assert mmd_loss(x, y).item() > mmd_loss(x, x.clone()).item()


def test_mmd_is_differentiable():
    x = torch.randn(64, 8, requires_grad=True)
    y = torch.randn(64, 8)
    mmd_loss(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_paraunitary_zero_for_unitary_matrix():
    """L_pu = 0 when each W is unitary (Eq. 25 satisfied exactly)."""
    q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.complex64))
    W = q.unsqueeze(0)  # L_net = 1
    val = paraunitary_loss(W)
    assert val.item() < 1e-5


def test_pauc_improvement():
    """L_pauc should decrease as targets get higher scores."""
    y = torch.cat([torch.ones(10), torch.zeros(100)])
    scores_bad = torch.cat([torch.ones(10) * 0.4, torch.ones(100) * 0.5])   # low targets
    scores_good = torch.cat([torch.ones(10) * 0.6, torch.ones(100) * 0.5])  # high targets
    val_bad = partial_auc_loss(scores_bad, y, alpha=0.1)
    val_good = partial_auc_loss(scores_good, y, alpha=0.1)
    assert val_good.item() < val_bad.item()


def test_composite_loss_weights():
    # Setup dummy data
    d = torch.rand(10, requires_grad=True)
    y = (torch.rand(10) > 0.5).long()
    L_sep = torch.tensor(0.5)
    W = torch.randn(2, 2, 2, dtype=torch.complex64)
    phi_syn = torch.randn(5, 8)
    phi_real = torch.randn(5, 8)

    criterion = DUCCEBMLoss(l1=1, l2=2, l3=3, l4=4, l5=5)
    total, parts = criterion(d, y, L_sep, W, phi_syn, phi_real)

    # total should equal l1*L_det + l2*L_sep + l3*L_pu + l4*L_da + l5*L_pauc
    manual_sum = (1 * parts["L_det"] + 2 * parts["L_sep"] +
                  3 * parts["L_pu"] + 4 * parts["L_da"] +
                  5 * parts["L_pauc"])
    assert torch.allclose(total, manual_sum)


# --------------------------------------------------------------------- #
# [ADDED DURING VERIFICATION] -- not proposed by Fable
# --------------------------------------------------------------------- #

def test_mmd_raises_on_degenerate_batch():
    """Fable's mmd_loss divides by n*(n-1) and m*(m-1) with no guard.
    n=1 or m=1 (which DOES occur in this codebase: e.g. a mini-batch of
    16 samples with a rare-class imbalance, or the 45-genuine-real-target
    pool referenced in MASTER_REAL_RESULTS_original_session.json split
    across many mini-batches) triggers a silent 0/0 -> NaN in the
    original code. The hardened version here raises a clear ValueError
    instead, so a NaN loss can never silently enter backprop."""
    x_ok = torch.randn(3, 4)
    x_bad = torch.randn(1, 4)
    with pytest.raises(ValueError):
        mmd_loss(x_ok, x_bad)
    with pytest.raises(ValueError):
        mmd_loss(x_bad, x_ok)


def test_k_degenerates_to_one_at_realistic_batch_sizes():
    """Quantitative demonstration of the batch-size/alpha mismatch flagged
    in the implementation prompt: at alpha=1e-3 and any mini-batch size
    used anywhere in this codebase (16..128, see CFG['batch'] and the
    ablation-suite batch=128), k_hard_negatives always returns 1, i.e.
    partial_auc_loss silently degrades to 'rank the single hardest
    clutter sample above every target sample' rather than truly
    representing the Pfa<=1e-3 region. This is the reason Stage 3 of the
    implementation plan requires accumulating scores over a buffer
    larger than one mini-batch before calling partial_auc_loss with the
    paper's alpha=1e-3."""
    for batch_size in (16, 32, 64, 128):
        # neg count is at most batch_size (usually less, since some rows
        # are positive), so this upper-bounds n_neg for every batch used
        # in this repository's CFG.
        assert k_hard_negatives(alpha=1e-3, n_neg=batch_size) == 1

    # Only once n_neg reaches the order of 1/alpha = 1000 does k start
    # growing past 1 -- confirming the fix direction described in the
    # implementation prompt (accumulate an epoch-scale buffer, not a
    # single mini-batch).
    assert k_hard_negatives(alpha=1e-3, n_neg=999) == 1
    assert k_hard_negatives(alpha=1e-3, n_neg=1500) == 2


def test_pauc_loss_is_finite_and_differentiable_with_k_equal_one():
    """Even in the degenerate k=1 regime demonstrated above, the loss
    itself must remain finite and backprop-able (a separate concern from
    whether it is a *good* pAUC estimator in that regime)."""
    torch.manual_seed(1)
    scores = torch.rand(32, requires_grad=True)
    y = (torch.rand(32) > 0.85).float()  # sparse positives, batch-realistic
    if y.sum() == 0:
        y[0] = 1.0
    loss = partial_auc_loss(scores, y, alpha=1e-3)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_composite_loss_skips_mmd_term_when_features_absent():
    """DUCCEBMLoss must not crash and must not silently add an L_da key
    when phi_syn/phi_real are not supplied (the default training path
    in this codebase, before Stage 4 MMD integration is switched on)."""
    d = torch.rand(10, requires_grad=True)
    y = (torch.rand(10) > 0.5).float()
    L_sep = torch.tensor(0.5)
    W = torch.randn(2, 2, 2, dtype=torch.complex64)
    criterion = DUCCEBMLoss()
    total, parts = criterion(d, y, L_sep, W, phi_syn=None, phi_real=None)
    assert "L_da" not in parts
    assert torch.isfinite(total)


# --------------------------------------------------------------------- #
# [ADDED IN STAGE 3] -- HardNegativeBuffer / partial_auc_loss_buffered,
# the batch-size/alpha fix required before this loss can be integrated
# into training (see FABLE_VERIFICATION_REPORT.md Sec. 2.1 and
# FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md Stage 3).
# --------------------------------------------------------------------- #

def test_buffer_k_grows_past_one_as_it_fills():
    """The whole point of the buffer: once enough mini-batches have been
    pushed, k should exceed 1 at alpha=1e-3, unlike the raw per-batch
    partial_auc_loss (see test_k_degenerates_to_one_at_realistic_batch_sizes
    above)."""
    from ducc_loss import HardNegativeBuffer, k_hard_negatives
    buf = HardNegativeBuffer(capacity=2000)
    torch.manual_seed(0)
    for _ in range(20):   # 20 mini-batches of 100 negatives = 2000 total
        buf.push(torch.rand(100))
    assert len(buf) == 2000
    assert k_hard_negatives(1e-3, len(buf)) == 2   # round(1e-3 * 2000) = 2


def test_buffer_respects_capacity_fifo():
    from ducc_loss import HardNegativeBuffer
    buf = HardNegativeBuffer(capacity=50)
    for i in range(5):
        buf.push(torch.full((20,), float(i)))   # batches of constant value i
    assert len(buf) == 50
    # only the last 2.5 "batches" worth of pushes should remain (FIFO)
    assert buf.scores.min().item() >= 2.0   # earliest batches (0, 1) evicted... partially


def test_partial_auc_loss_buffered_matches_manual_computation():
    from ducc_loss import HardNegativeBuffer, partial_auc_loss_buffered, k_hard_negatives
    torch.manual_seed(1)
    buf = HardNegativeBuffer(capacity=1000)
    pos = torch.rand(4, requires_grad=True) * 0.3 + 0.6   # high scores
    neg = torch.rand(50) * 0.4                             # low scores, current batch
    loss = partial_auc_loss_buffered(pos, neg, buf, alpha=1e-3, tau=0.1)

    # manual recomputation, replicating the function's own logic
    k = k_hard_negatives(1e-3, buf.scores.numel())
    hard_neg, _ = torch.topk(buf.scores, min(k, buf.scores.numel()))
    diff = pos.detach().unsqueeze(1) - hard_neg.unsqueeze(0)
    expected = 1.0 - torch.sigmoid(diff / 0.1).mean()
    assert torch.allclose(loss.detach(), expected, atol=1e-6)


def test_partial_auc_loss_buffered_is_differentiable_and_gradients_flow_to_positives():
    from ducc_loss import HardNegativeBuffer, partial_auc_loss_buffered
    torch.manual_seed(2)
    buf = HardNegativeBuffer(capacity=500)
    for _ in range(5):
        buf.push(torch.rand(50))    # warm up the buffer with past (detached) batches

    pos = torch.rand(8, requires_grad=True)
    neg_current = torch.rand(20, requires_grad=True)
    loss = partial_auc_loss_buffered(pos, neg_current, buf, alpha=1e-3)
    loss.backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    # neg_current's grad may legitimately be all-zero if none of its
    # entries were selected among the hardest-k this call -- only check
    # it doesn't error and is finite where defined.
    assert neg_current.grad is None or torch.isfinite(neg_current.grad).all()


def test_partial_auc_loss_buffered_improves_as_positives_separate_from_hard_negatives():
    """Same spirit as test_pauc_improvement above, but for the buffered
    version, confirmed on a buffer large enough for k>1 (alpha=1e-3,
    n_neg=2000 -> k=2, per test_buffer_k_grows_past_one_as_it_fills)."""
    from ducc_loss import HardNegativeBuffer, partial_auc_loss_buffered
    torch.manual_seed(3)
    buf_bad = HardNegativeBuffer(capacity=2000)
    buf_good = HardNegativeBuffer(capacity=2000)
    neg_fixed = torch.rand(2000) * 0.5     # identical hard-negative population for both
    buf_bad.push(neg_fixed.clone())
    buf_good.push(neg_fixed.clone())

    pos_bad = torch.full((10,), 0.45)      # barely above the negative pool
    pos_good = torch.full((10,), 0.9)      # clearly separated
    loss_bad = partial_auc_loss_buffered(pos_bad, torch.rand(1) * 0.1, buf_bad, alpha=1e-3)
    loss_good = partial_auc_loss_buffered(pos_good, torch.rand(1) * 0.1, buf_good, alpha=1e-3)
    assert loss_good.item() < loss_bad.item()


# --------------------------------------------------------------------- #
# [ADDED IN STAGE 4] -- regression test for a real bug caught by running
# the MMD-integration smoke test in du_ccebm_simulation.py: DUCCEBM's
# return_features=True path returns a 5-tuple
# (score, z_re, z_im, pu_loss, feats), and the FIRST integration attempt
# mis-unpacked it as a 2-tuple (`_, phi_real = model(...)`), raising
# ValueError at the first MMD-enabled training step. This test pins the
# correct 5-tuple contract so a future refactor cannot silently
# reintroduce the same mismatch.
# --------------------------------------------------------------------- #

def test_return_features_contract_is_a_five_tuple():
    import sys
    sys.path.insert(0, '.')
    import du_ccebm_simulation as sim
    torch.manual_seed(0)
    model = sim.DUCCEBM(L_net=1, N=2, L_filt=2, K_white=2)
    x = torch.randn(3, 2, 2, 8)
    out = model(x, return_features=True)
    assert len(out) == 5, f"expected 5-tuple, got {len(out)}"
    score, z_re, z_im, pu_loss, feats = out   # must not raise
    assert feats.shape == (3, 4)
    assert score.shape == (3,)
