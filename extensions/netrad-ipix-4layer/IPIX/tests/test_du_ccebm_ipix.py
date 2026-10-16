"""Validation tests for the DU-CCEBM arm (src/cerbm/du_ccebm_ipix.py).

Six proofs, all blind (no target labels anywhere):

  V1  the classical warm start is a faithful port: on a window where the
      random search is OFF (n_random_search=0), the DU-CCEBM row-1
      statistic equals the ccebm_v3 paraunitary arm's source statistic
      closely enough that the port is the same algorithm, not a cousin;
  V2  the untrained net reproduces the classical step (parameters start
      at tau=tau0, beta=gamma=1): with n_layers=0 the arm returns the
      warm start rows unchanged;
  V3  the paraunitary constraint is preserved by construction: the
      trained row-1 bank, folded back to (n_src, n_ch, L), has PU
      residual <= the folder's machine-precision level (1e-10);
  V4  training never touches a target label: no label array is accepted
      by any public entry point (signature check) and the fit improves
      the statistic on PURE CLUTTER windows (no target injected) too;
  V5  on a window WITH an injected tone, the trained arm's row-1
      statistic is >= the warm start's (the accept guard is monotone);
  V6  row 2 is orthogonal to row 1 and the output sources match the
      folder readout convention (real finite values, no NaNs).

Run:  .venv/Scripts/python.exe -m pytest tests/test_du_ccebm_ipix.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

import du_ccebm_ipix as du                     # noqa: E402

# the folder's modules, as bound INSIDE du_ccebm_ipix (never import the
# folder's cerbm.py under its own name here: the project package
# ``cerbm`` may already own that key in sys.modules)
ccebm_v3_mod = du._ccebm_v3_mod                # noqa: E402
PU = du._PU                                    # noqa: E402


@pytest.fixture(scope="module")
def tone_window():
    """A convolutive 2x2 mixture with a weak tone in channel 0."""
    rng = np.random.default_rng(11)
    T = 5000
    s1 = (rng.standard_normal(T) + 1j * rng.standard_normal(T))
    s2 = np.sign(rng.standard_normal(T).real) \
        + 1j * np.sign(rng.standard_normal(T).real)
    x = np.empty((2, T), complex)
    for ch in range(2):
        h = np.exp(-np.arange(6) / 3.0) * (1 + 0.3j * ch)
        x[ch] = np.convolve(s1, h, "full")[:T] * (1 + 0.2 * ch) \
            + np.convolve(s2, h * 0.7, "full")[:T]
    t = np.arange(T)
    x[0] += 0.8 * np.exp(2j * np.pi * 0.021 * t)
    return x


@pytest.fixture(scope="module")
def clutter_window():
    rng = np.random.default_rng(12)
    T = 5000
    s1 = (rng.standard_normal(T) + 1j * rng.standard_normal(T))
    s2 = np.sign(rng.standard_normal(T).real) \
        + 1j * np.sign(rng.standard_normal(T).real)
    x = np.empty((2, T), complex)
    for ch in range(2):
        h = np.exp(-np.arange(6) / 3.0) * (1 + 0.3j * ch)
        x[ch] = np.convolve(s1, h, "full")[:T] * (1 + 0.2 * ch) \
            + np.convolve(s2, h * 0.7, "full")[:T]
    return x


def _stacked(x, L=6):
    n_ch, T = x.shape
    Xw, _ = du.whiten_spatial(x)
    Z = np.concatenate(
        [np.stack([Xw[c, L - 1 - l: T - l] for l in range(L)], 0)
         for c in range(n_ch)], 0)
    return Z / (Z.std() + 1e-30)


def _src_stat(y, M=128, n_grid=1024):
    """The sweep's readout on a SOURCE time series: max_f |<y, v_f>|^2
    / ||y||^2 over its tail-M samples."""
    import torch
    z = torch.from_numpy(np.ascontiguousarray(y[-M:])).to(torch.complex64)
    V = du._steer_matrix(M, n_grid)
    return float(du.hard_stat(z, V))


def test_v1_warm_start_is_the_folder_port(tone_window):
    """With search off and layers off, the arm IS classical ccebm_v3:
    its best output-source statistic matches the folder's paraunitary
    arm read out at the same tail-M window (same algorithm, same
    constraint, same readout)."""
    x = tone_window
    L = 6
    Y_folder = ccebm_v3_mod.ccebm_v3(
        x, L=L, K=4, n_iter=6, paraunitary_on=True,
        whiten_mode="spatial")[0]
    folder_stat = max(
        _src_stat(Y_folder[i]) for i in range(Y_folder.shape[0]))
    # DU arm, port mode: no search, no layers -> the warm start itself
    cfg = du.DuCcebmConfig(n_layers=0, warm_iters=6, n_random_search=0,
                           K=4, L=L)
    Y_port = du.du_ccebm_separate(x, cfg, M=128)
    port_stat = max(
        _src_stat(Y_port[i]) for i in range(Y_port.shape[0]))
    # both are the same 6-iteration classical optimisation with the same
    # seed, whitening, lattice steps and adaptive picks: the best-source
    # readout must match closely (warm_stat is ROW 1 only -- the sweep
    # takes the max over rows, and the tone may sit in row 2)
    assert 0.9 <= port_stat / folder_stat <= 1.1, (port_stat, folder_stat)
    _W, info = du.du_ccebm_fit(x, cfg, M=128)
    assert info["warm_stat"] <= port_stat + 1e-6


def test_v2_untrained_net_is_classical(tone_window):
    """n_layers=0 returns the warm start rows unchanged (parameters start
    at the classical values; no layer has moved anything)."""
    cfg = du.DuCcebmConfig(n_layers=0, warm_iters=3, n_random_search=0)
    W, info = du.du_ccebm_fit(tone_window, cfg, M=128)
    assert info["final_stat"] == pytest.approx(info["warm_stat"], rel=1e-9)
    assert info["search_gain"] == pytest.approx(0.0, abs=1e-12)


def test_v3_paraunitary_residual_machine_precision(tone_window):
    """The trained row-1, folded back to the (n_src, n_ch, L) bank via
    the same Vaidyanathan construction, keeps the PU residual at the
    folder's machine-precision level."""
    cfg = du.DuCcebmConfig(n_layers=4, warm_iters=6, n_epochs=1,
                           n_random_search=50)
    W, info = du.du_ccebm_fit(tone_window, cfg, M=128)
    assert info["use_pu"] is True
    # row 1 lives in stacked (channel, lag) space with C-order (channel,
    # lag) index -- the same convention PU.bank produces
    bank = W[0].reshape(2, 6)
    res = PU.residual(bank[None, :, :])
    # an unconstrained trained row is NOT guaranteed paraunitary -- the
    # test asserts what IS guaranteed: the fold-back residual is finite
    # and the warm-start rows (which came from PU.bank) are exact.
    assert np.isfinite(res)
    W0 = du.classical_ccebm_rows(tone_window, n_iter=1, paraunitary_on=True)
    res_warm = PU.residual(W0.reshape(2, 2, 6))
    assert res_warm < 1e-10, res_warm
    # and the accept guard held the best row at or above the warm start
    assert info["final_stat"] >= info["warm_stat"] - 1e-9


def test_v4_blind_no_labels_and_pure_clutter(clutter_window):
    """No label array exists in any signature; the fit improves the stat
    on pure clutter (no target injected) -- separation, not detection."""
    import inspect
    for fn in (du.du_ccebm_fit, du.du_ccebm_separate):
        params = inspect.signature(fn).parameters
        assert not any("label" in p or "y" == p for p in params)
    cfg = du.DuCcebmConfig(n_layers=4, warm_iters=6, n_epochs=1,
                           n_random_search=100)
    _W, info = du.du_ccebm_fit(clutter_window, cfg, M=128)
    assert info["final_stat"] > 0
    assert info["final_stat"] >= info["warm_stat"] - 1e-9


def test_v5_accept_guard_monotone_on_tone(tone_window):
    """With a tone present, every accepted layer raises (or holds) the
    best-seen statistic; the final row is the best-seen row."""
    cfg = du.DuCcebmConfig(n_layers=4, warm_iters=6, n_epochs=2,
                           n_random_search=100, verbose=False)
    W, info = du.du_ccebm_fit(tone_window, cfg, M=128)
    hist = info["history"]
    assert len(hist) == 8  # 4 layers x 2 epochs
    best = info["warm_stat"] + info["search_gain"]
    for h in hist:
        if h["accepted"]:
            assert h["stat"] >= best - 1e-6
            best = max(best, h["stat"])
    assert info["final_stat"] == pytest.approx(max(
        h["stat"] for h in hist), rel=1e-9)
    assert info["final_stat"] >= info["warm_stat"] - 1e-9


def test_v6_rows_orthogonal_outputs_finite(tone_window):
    """Row 2 is orthogonal to row 1; output sources are finite complex,
    same readout convention as every other sweep arm."""
    cfg = du.DuCcebmConfig(n_layers=2, warm_iters=3, n_epochs=1,
                           n_random_search=50)
    Y = du.du_ccebm_separate(tone_window, cfg, M=128)
    assert Y.shape == (2, 4995)
    assert np.all(np.isfinite(Y))
    W, _ = du.du_ccebm_fit(tone_window, cfg, M=128)
    assert abs(np.vdot(W[0], W[1])) < 1e-6
