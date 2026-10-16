"""Regression tests for the new vectorized CCEBM and JSON-driven paper tables."""

from pathlib import Path

import numpy as np
import pytest

import du_ccebm_simulation as sim
from classical_ref.optimized_ccebm import run_batched_ccebm


ROOT = Path(__file__).resolve().parent


def test_vectorized_ccebm_matches_scalar_nu2():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(3, 2, 128)) + 1j * rng.normal(size=(3, 2, 128))
    batched = run_batched_ccebm(
        x, L=8, K=5, max_iter=50, tol=1e-5, prefer_cuda=False
    )
    scalar = sim.ClassicalCCEBM(L=8, K=5, max_iter=50, tol=1e-5)
    expected = []
    for row in x:  # test oracle only; production implementation has no window loop
        _, z2, _ = scalar.fit_separate(row[0], row[1])
        expected.append(sim.k_distribution_mom(z2)[0])
    assert np.allclose(batched.nu2, expected, rtol=1e-9, atol=1e-9)


def test_paper_tables_equal_generated_json():
    json_path = ROOT / "results" / "REVISION_REAL_RESULTS_v2.json"
    if not json_path.exists():
        pytest.skip("revision-v2 experiments have not completed yet")
    from scripts.render_paper_from_json import render_expected_fragments

    if not (ROOT / "paper" / "du_ccebm_taes.tex").is_file():
        pytest.skip("historical manuscript deliberately excluded from public software-only export")
    fragments = render_expected_fragments(json_path)
    tex = (ROOT / "paper" / "du_ccebm_taes.tex").read_text(encoding="utf-8")
    for fragment in fragments.values():
        assert fragment in tex
