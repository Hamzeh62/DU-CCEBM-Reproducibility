"""Render the revision-v2 LaTeX result blocks from the generated JSON.

The same functions are imported by the unit test, making the JSON the single
source of truth for every newly reported numeric table value.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results" / "REVISION_REAL_RESULTS_v2.json"
TEX_PATH = ROOT / "paper" / "du_ccebm_taes.tex"


def _fmt(value: float, digits=3) -> str:
    return f"{value:.{digits}f}"


def _signed(value: float, digits=3) -> str:
    return f"{value:+.{digits}f}"


def _metric(entry: dict) -> str:
    return f"{_fmt(entry['point'])} [{_fmt(entry['ci95'][0])}, {_fmt(entry['ci95'][1])}]"


def render_expected_fragments(path: Path = JSON_PATH) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    methods = data["E1_full_classical_and_paired_bootstrap"]["inference"]["methods"]
    deltas = data["E1_full_classical_and_paired_bootstrap"]["inference"][
        "paired_auc_differences"
    ]
    bench = data["E2_wall_clock"]
    lofo = data["E3_leave_one_file_out"]
    e4 = data["E4_operating_points"]["methods"]
    e5 = data["E5_mixed_sign_flipped_anmf"]
    mean_params = [
        row for row in data["E6_layer_parameters"]["rows"] if row["component"] == "mean"
    ]

    genuine_rows = "\n".join(
        f"{name} & {_metric(methods[name]['auc'])} & {_metric(methods[name]['pd_at_pfa_1e-2'])}\\\\"
        for name in ("DU-CCEBM", "Classical CCEBM", "ANMF", "CA-CFAR")
    )
    delta_rows = "\n".join(
        f"{key.replace('_minus_', ' $-$ ')} & {_signed(value['point'])} "
        f"[{_signed(value['ci95'][0])}, {_signed(value['ci95'][1])}]\\\\"
        for key, value in deltas.items()
    )
    benchmark_rows = "\n".join(
        f"{name} & {_fmt(bench[name]['mean_ms_per_window'], 3)}\\\\"
        for name in ("DU-CCEBM", "Classical CCEBM", "ANMF", "CA-CFAR")
    )
    operating_rows = "\n".join(
        f"{name} & {_fmt(e4[name]['auc']['point'])} & "
        f"{_metric(e4[name]['pd_at_pfa_1e-2'])} & "
        f"{_metric(e4[name]['pd_at_pfa_1e-3'])} & "
        f"{_fmt(e4[name]['normalized_auc_far_1e-3_to_1e-1']['point'])}\\\\"
        for name in ("DU-CCEBM", "Classical CCEBM", "ANMF", "CA-CFAR")
    )
    lofo_rows = "\n".join(
        f"{row['held_out_file'].replace('_', r'\_')} & {row['n_target']} & {row['n_clutter']} & "
        f"{_fmt(row['auc_DU-CCEBM'])}\\\\"
        for row in lofo["per_file"]
    )
    params_rows = "\n".join(
        f"{row['layer']} & {_fmt(row['beta1'])} & {_fmt(row['beta2'])} & "
        f"{_fmt(row['a'])} & {_fmt(row['tau'])} & {_fmt(row['gamma'])}\\\\"
        for row in mean_params
    )
    return {
        "REV2_GENUINE_ROWS": genuine_rows,
        "REV2_DELTA_ROWS": delta_rows,
        "REV2_BENCHMARK_ROWS": benchmark_rows,
        "REV2_OPERATING_ROWS": operating_rows,
        "REV2_LOFO_ROWS": lofo_rows,
        "REV2_PARAMS_ROWS": params_rows,
        "REV2_LOFO_SUMMARY": (
            f"{_fmt(lofo['DU-CCEBM_mean_auc'])} $\\pm$ {_fmt(lofo['DU-CCEBM_std_auc'])}"
        ),
        "REV2_SIGNFLIP": _fmt(e5["ANMF_auc_sign_flipped"]),
        "REV2_CCEBM_RUNTIME": _fmt(
            data["E1_full_classical_and_paired_bootstrap"]["classical_run"]["wall_clock_s"],
            1,
        ),
    }


def replace_markers(tex: str, fragments: dict[str, str]) -> str:
    for name, value in fragments.items():
        pattern = rf"(?s)(% BEGIN {name}\n).*?(\n% END {name})"
        replacement = rf"\g<1>{value}\g<2>"
        tex, n = re.subn(
            pattern,
            lambda match: match.group(1) + value + match.group(2),
            tex,
        )
        if n != 1:
            raise RuntimeError(f"expected exactly one marker block for {name}, found {n}")
    return tex


def main():
    fragments = render_expected_fragments()
    original = TEX_PATH.read_text(encoding="utf-8")
    updated = replace_markers(original, fragments)
    TEX_PATH.write_text(updated, encoding="utf-8")
    shutil.copy2(TEX_PATH, ROOT / "outputs" / TEX_PATH.name)


if __name__ == "__main__":
    main()
