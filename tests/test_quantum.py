import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum import (  # noqa: E402
    MAX_OPERATIONS,
    Operation,
    apply_gate,
    basis_labels,
    bloch_vector,
    cnot_matrix,
    initial_state,
    probabilities,
    run_circuit,
    sample_counts,
    total_variation_distance,
    validate_state,
)

S = 1 / math.sqrt(2)


def assert_state(actual, expected, atol=1e-12):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected, dtype=complex), atol=atol)


def test_initial_states_and_basis_order():
    assert_state(initial_state(1), [1, 0])
    assert_state(initial_state(2), [1, 0, 0, 0])
    assert basis_labels(1) == ["|0⟩", "|1⟩"]
    assert basis_labels(2) == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]


def test_x_maps_zero_to_one():
    assert_state(apply_gate(initial_state(1), "X", 0, 1), [0, 1])


def test_h_gives_equal_probabilities():
    state = apply_gate(initial_state(1), "H", 0, 1)
    assert_state(state, [S, S])
    np.testing.assert_allclose(probabilities(state), [0.5, 0.5], atol=1e-12)


def test_h_twice_returns_initial_state():
    assert_state(run_circuit([Operation("H", 0), Operation("H", 0)], 1), [1, 0])


def test_hzh_maps_zero_to_one():
    ops = [Operation("H", 0), Operation("Z", 0), Operation("H", 0)]
    assert_state(run_circuit(ops, 1), [0, 1])


@pytest.mark.parametrize(
    "theta, expected",
    [
        (0, [1, 0]),
        (90, [S, S]),
        (180, [0, 1]),
        (-90, [S, -S]),
        (360, [-1, 0]),
        (60, [math.cos(math.pi / 6), math.sin(math.pi / 6)]),
    ],
)
def test_ry_known_angles(theta, expected):
    assert_state(apply_gate(initial_state(1), "RY", 0, 1, theta_deg=theta), expected)


def test_single_qubit_gate_respects_qubit_order():
    # X на q0 меняет левый бит: |00⟩ -> |10⟩; X на q1 — правый: |00⟩ -> |01⟩.
    assert_state(apply_gate(initial_state(2), "X", 0, 2), [0, 0, 1, 0])
    assert_state(apply_gate(initial_state(2), "X", 1, 2), [0, 1, 0, 0])


@pytest.mark.parametrize(
    "control, target, mapping",
    [
        (0, 1, {0: 0, 1: 1, 2: 3, 3: 2}),  # |10⟩ <-> |11⟩
        (1, 0, {0: 0, 1: 3, 2: 2, 3: 1}),  # |01⟩ <-> |11⟩
    ],
)
def test_cnot_both_directions(control, target, mapping):
    for source, destination in mapping.items():
        basis = np.zeros(4, dtype=complex)
        basis[source] = 1
        result = apply_gate(basis, "CNOT", target, 2, control=control)
        expected = np.zeros(4, dtype=complex)
        expected[destination] = 1
        assert_state(result, expected)
    matrix = cnot_matrix(control, target)
    np.testing.assert_allclose(matrix @ matrix, np.eye(4), atol=1e-12)


def test_bell_state_amplitudes():
    state = run_circuit([Operation("H", 0), Operation("CNOT", target=1, control=0)], 2)
    assert_state(state, [S, 0, 0, S])


def test_bell_state_with_reversed_control():
    state = run_circuit([Operation("H", 1), Operation("CNOT", target=0, control=1)], 2)
    assert_state(state, [S, 0, 0, S])


@pytest.mark.parametrize("n", [1, 2])
def test_normalization_preserved_and_probabilities_sum_to_one(n):
    rng = np.random.default_rng(0)
    gates = ["X", "H", "Z", "RY"] + (["CNOT"] if n == 2 else [])
    ops = []
    for _ in range(MAX_OPERATIONS):
        gate = gates[rng.integers(len(gates))]
        target = int(rng.integers(n))
        if gate == "CNOT":
            ops.append(Operation("CNOT", target=target, control=1 - target))
        elif gate == "RY":
            ops.append(Operation("RY", target, theta_deg=float(rng.uniform(-360, 360))))
        else:
            ops.append(Operation(gate, target))
    state = run_circuit(ops, n)
    assert abs(np.linalg.norm(state) - 1) < 1e-12
    assert math.isclose(probabilities(state).sum(), 1.0, abs_tol=1e-12)


def test_invalid_state_is_rejected_not_normalized():
    with pytest.raises(ValueError):
        validate_state(np.array([1, 1], dtype=complex), 1)
    with pytest.raises(ValueError):
        apply_gate(np.array([1, 1], dtype=complex), "X", 0, 1)
    with pytest.raises(ValueError):
        validate_state(np.array([1, 0, 0], dtype=complex), 1)


def test_seeded_sampling_is_reproducible():
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    a = sample_counts(probs, 1000, 123)
    b = sample_counts(probs, 1000, 123)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, sample_counts(probs, 1000, 124))


@pytest.mark.parametrize("shots", [1, 10, 777, 10_000])
def test_counts_sum_to_shots(shots):
    assert sample_counts(np.array([0.5, 0.5]), shots, 1).sum() == shots


def test_impossible_outcomes_get_zero_counts():
    bell = run_circuit([Operation("H", 0), Operation("CNOT", target=1, control=0)], 2)
    counts = sample_counts(probabilities(bell), 10_000, 5)
    assert counts[1] == 0 and counts[2] == 0
    hh = run_circuit([Operation("H", 0), Operation("H", 0)], 1)
    assert sample_counts(probabilities(hh), 10_000, 5)[1] == 0


def test_frequencies_close_but_not_required_equal():
    counts = sample_counts(np.array([0.5, 0.5]), 10_000, 2024)
    assert abs(counts[0] / 10_000 - 0.5) < 0.03


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gate="Y", target=0, n_qubits=1),
        dict(gate="X", target=1, n_qubits=1),
        dict(gate="X", target=-1, n_qubits=2),
        dict(gate="CNOT", target=0, n_qubits=1, control=0),
        dict(gate="CNOT", target=1, n_qubits=2, control=1),
        dict(gate="CNOT", target=1, n_qubits=2),
        dict(gate="RY", target=0, n_qubits=1),
        dict(gate="RY", target=0, n_qubits=1, theta_deg=float("nan")),
        dict(gate="H", target=0, n_qubits=1, theta_deg=30.0),
        dict(gate="H", target=0, n_qubits=3),
    ],
)
def test_invalid_gate_parameters(kwargs):
    n = kwargs.pop("n_qubits")
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    with pytest.raises(ValueError):
        apply_gate(state, n_qubits=n, **kwargs)


def test_circuit_length_limit():
    with pytest.raises(ValueError):
        run_circuit([Operation("X", 0)] * (MAX_OPERATIONS + 1), 1)


@pytest.mark.parametrize("shots, seed", [(0, 1), (10, -1), (10, 2**32), (2.5, 1)])
def test_invalid_sampling_parameters(shots, seed):
    with pytest.raises(ValueError):
        sample_counts(np.array([0.5, 0.5]), shots, seed)


def test_sampling_rejects_invalid_distribution():
    with pytest.raises(ValueError):
        sample_counts(np.array([0.7, 0.7]), 10, 1)


def test_total_variation_distance():
    assert total_variation_distance([0.5, 0.5], [0.5, 0.5]) == 0
    assert math.isclose(total_variation_distance([1, 0], [0, 1]), 1.0)
    assert math.isclose(total_variation_distance([0.6, 0.4], [0.5, 0.5]), 0.1)
    assert math.isclose(
        total_variation_distance([0.4, 0.1, 0.3, 0.2], [0.25] * 4), 0.5 * (0.15 + 0.15 + 0.05 + 0.05)
    )
    with pytest.raises(ValueError):
        total_variation_distance([1.0], [0.5, 0.5])


@pytest.mark.parametrize(
    "state, expected",
    [
        ([1, 0], (0, 0, 1)),
        ([0, 1], (0, 0, -1)),
        ([S, S], (1, 0, 0)),
        ([S, -S], (-1, 0, 0)),
        ([S, 1j * S], (0, 1, 0)),
    ],
)
def test_bloch_vector(state, expected):
    np.testing.assert_allclose(bloch_vector(np.array(state, dtype=complex)), expected, atol=1e-12)
