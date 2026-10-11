import codecs
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_view import circuit_svg, format_state  # noqa: E402
from experiments import (  # noqa: E402
    BELL,
    BIT_FLIP,
    CONVERGENCE_SEEDS,
    CONVERGENCE_SHOTS,
    INTERFERENCE_HH,
    INTERFERENCE_HZH,
    SUPERPOSITION,
    run_convergence_experiment,
    run_preset,
)
from quantum import Operation, run_circuit, total_variation_distance  # noqa: E402
from reporting import (  # noqa: E402
    Measurement,
    config_from_json,
    config_to_json,
    html_report,
    measurements_csv,
)

S = 1 / math.sqrt(2)


def test_bit_flip_preset():
    state, probs = run_preset(BIT_FLIP)
    np.testing.assert_allclose(state, [0, 1], atol=1e-12)
    np.testing.assert_allclose(probs, [0, 1], atol=1e-12)


def test_superposition_preset():
    _, probs = run_preset(SUPERPOSITION)
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


def test_interference_presets():
    _, p_hh = run_preset(INTERFERENCE_HH)
    _, p_hzh = run_preset(INTERFERENCE_HZH)
    np.testing.assert_allclose(p_hh, [1, 0], atol=1e-12)
    np.testing.assert_allclose(p_hzh, [0, 1], atol=1e-12)


def test_bell_preset():
    state, probs = run_preset(BELL)
    np.testing.assert_allclose(state, [S, 0, 0, S], atol=1e-12)
    np.testing.assert_allclose(probs, [0.5, 0, 0, 0.5], atol=1e-12)


def test_convergence_experiment_structure_and_reproducibility():
    first = run_convergence_experiment()
    second = run_convergence_experiment()
    assert first.runs == second.runs
    assert len(first.runs) == len(CONVERGENCE_SHOTS) * len(CONVERGENCE_SEEDS)
    assert [row["shots"] for row in first.summary] == list(CONVERGENCE_SHOTS)
    for row in first.runs:
        counts = row["count |0⟩"] + row["count |1⟩"]
        assert counts == row["shots"]
        freqs = [row["count |0⟩"] / row["shots"], row["count |1⟩"] / row["shots"]]
        assert math.isclose(row["tvd"], total_variation_distance(freqs, [0.5, 0.5]), abs_tol=1e-12)
    means = [row["mean_tvd"] for row in first.summary]
    assert means[-1] < means[0]
    assert 0 <= first.non_monotonic_series <= len(CONVERGENCE_SEEDS)


def test_convergence_rejects_bad_input():
    with pytest.raises(ValueError):
        run_convergence_experiment(shots_list=(100, 10))


def test_config_json_roundtrip():
    ops = [Operation("H", 0), Operation("RY", 1, theta_deg=45.0), Operation("CNOT", target=0, control=1)]
    text = config_to_json(ops, 2)
    data = json.loads(text)
    assert data["basis_order"] == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    assert config_from_json(text) == (2, ops)


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"format": "other"}),
        json.dumps({"format": "quantumlab-circuit", "n_qubits": 1,
                    "operations": [{"gate": "CNOT", "target": 0, "control": 1}]}),
        json.dumps({"format": "quantumlab-circuit", "n_qubits": 3, "operations": []}),
    ],
)
def test_config_rejects_invalid(payload):
    with pytest.raises(ValueError):
        config_from_json(payload)


def test_measurements_csv_has_bom_and_values():
    state = run_circuit(BELL.operations, 2)
    m = Measurement(shots=100, seed=3, counts=np.array([52, 0, 0, 48]))
    data = measurements_csv(state, 2, m)
    assert data.startswith(codecs.BOM_UTF8)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert rows[0][0] == "Базисное состояние"
    assert [r[0] for r in rows[1:]] == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    assert rows[1][4] == "52" and float(rows[1][5]) == 0.52


def test_html_report_escapes_and_contains_data():
    state = run_circuit(BELL.operations, 2)
    m = Measurement(shots=100, seed=3, counts=np.array([52, 0, 0, 48]))
    report = html_report(list(BELL.operations), 2, state, m)
    assert "<script" not in report
    assert "|00⟩" in report and "0.707107" in report and "seed" in report
    assert "классическом компьютере" in report


def test_svg_diagram_matches_operations():
    ops = [Operation("H", 0), Operation("CNOT", target=1, control=0), Operation("RY", 1, theta_deg=30.0)]
    svg = circuit_svg(ops, 2)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count('r="5"') == 1  # одна управляющая точка CNOT
    assert "RY" in svg and "30°" in svg
    with pytest.raises(ValueError):
        circuit_svg([Operation("CNOT", target=1, control=0)], 1)


def test_format_state():
    assert format_state(np.array([S, 0, 0, S]), 2) == "0.7071|00⟩ + 0.7071|11⟩"
    assert format_state(np.array([S, -S]), 1) == "0.7071|0⟩ − 0.7071|1⟩"
