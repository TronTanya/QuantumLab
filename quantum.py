"""Ядро QuantumLab: моделирование вектора состояния для одного или двух кубитов.

Соглашение о базисе: метки |q0 q1⟩, q0 — левый (старший) бит, q1 — правый.
Порядок базиса для двух кубитов: |00⟩, |01⟩, |10⟩, |11⟩, т.е. индекс = 2·q0 + q1.
Все вычисления выполняются на классическом компьютере.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

MODEL_VERSION = "1.0.0"
MAX_QUBITS = 2
MAX_OPERATIONS = 20
NORM_TOLERANCE = 1e-9
MIN_SHOTS = 1
MAX_SHOTS = 1_000_000
MAX_SEED = 2**32 - 1

SINGLE_QUBIT_GATES = ("X", "H", "Z", "RY")
TWO_QUBIT_GATES = ("CNOT",)
SUPPORTED_GATES = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES

BASIS_CONVENTION = (
    "Метки |q0 q1⟩: q0 — левый бит, q1 — правый бит. "
    "Порядок базиса: |00⟩, |01⟩, |10⟩, |11⟩ (индекс = 2·q0 + q1)."
)

_SQRT1_2 = 1.0 / math.sqrt(2.0)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class Operation:
    """Одна операция схемы.

    gate: "X", "H", "Z", "RY" или "CNOT".
    target: целевой кубит.
    control: управляющий кубит (только для CNOT).
    theta_deg: угол в градусах (только для RY).
    """

    gate: str
    target: int
    control: int | None = None
    theta_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"gate": self.gate, "target": self.target}
        if self.control is not None:
            data["control"] = self.control
        if self.theta_deg is not None:
            data["theta_deg"] = self.theta_deg
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Operation":
        if not isinstance(data, dict):
            raise ValueError("Операция должна быть объектом JSON.")
        try:
            gate = str(data["gate"]).upper()
            target = data["target"]
        except KeyError as exc:
            raise ValueError(f"В описании операции нет поля {exc.args[0]!r}.") from exc
        control = data.get("control")
        theta = data.get("theta_deg")
        return Operation(
            gate=gate,
            target=_as_int(target, "target"),
            control=None if control is None else _as_int(control, "control"),
            theta_deg=None if theta is None else float(theta),
        )


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Поле {name} должно быть целым числом.")
    return int(value)


def _check_n_qubits(n_qubits: int) -> None:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError("Число кубитов должно быть целым.")
    if not 1 <= n_qubits <= MAX_QUBITS:
        raise ValueError(f"Поддерживается только 1 или {MAX_QUBITS} кубита.")


def _check_qubit(index: int, n_qubits: int, name: str) -> None:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"Номер кубита ({name}) должен быть целым.")
    if not 0 <= index < n_qubits:
        raise ValueError(
            f"Кубит q{index} ({name}) не существует в схеме из {n_qubits} кубит(ов)."
        )


def basis_labels(n_qubits: int) -> list[str]:
    """Метки базисных состояний в порядке индексов: |0⟩,|1⟩ или |00⟩..|11⟩."""
    _check_n_qubits(n_qubits)
    return [f"|{i:0{n_qubits}b}⟩" for i in range(2**n_qubits)]


def initial_state(n_qubits: int) -> np.ndarray:
    """Начальное состояние |0⟩ или |00⟩."""
    _check_n_qubits(n_qubits)
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def state_norm(state: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(state) ** 2)))


def validate_state(state: np.ndarray, n_qubits: int, tol: float = NORM_TOLERANCE) -> None:
    """Проверяет размерность и нормировку. Не нормирует молча — бросает ошибку."""
    _check_n_qubits(n_qubits)
    arr = np.asarray(state)
    if arr.shape != (2**n_qubits,):
        raise ValueError(
            f"Ожидался вектор длины {2**n_qubits}, получена форма {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Вектор состояния содержит нечисловые значения.")
    norm = state_norm(arr)
    if abs(norm - 1.0) > tol:
        raise ValueError(f"Состояние не нормировано: ‖ψ‖ = {norm:.12f}.")


def ry_matrix(theta_rad: float) -> np.ndarray:
    """RY(θ) = [[cos θ/2, −sin θ/2], [sin θ/2, cos θ/2]]."""
    if not math.isfinite(theta_rad):
        raise ValueError("Угол поворота должен быть конечным числом.")
    c, s = math.cos(theta_rad / 2.0), math.sin(theta_rad / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def single_qubit_matrix(gate: str, theta_deg: float | None = None) -> np.ndarray:
    gate = gate.upper()
    if gate == "X":
        return _X.copy()
    if gate == "H":
        return _H.copy()
    if gate == "Z":
        return _Z.copy()
    if gate == "RY":
        if theta_deg is None:
            raise ValueError("Для RY необходимо указать угол в градусах.")
        return ry_matrix(math.radians(theta_deg))
    raise ValueError(f"Неизвестный однокубитный гейт: {gate!r}.")


def expand_single_qubit(matrix: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Оператор на полном пространстве. q0 — левый множитель в тензорном произведении."""
    _check_n_qubits(n_qubits)
    _check_qubit(target, n_qubits, "target")
    if matrix.shape != (2, 2):
        raise ValueError("Однокубитная матрица должна иметь размер 2×2.")
    full = np.array([[1]], dtype=complex)
    for q in range(n_qubits):
        full = np.kron(full, matrix if q == target else _I)
    return full


def cnot_matrix(control: int, target: int, n_qubits: int = 2) -> np.ndarray:
    """Матрица CNOT как перестановка базисных состояний."""
    _check_n_qubits(n_qubits)
    if n_qubits < 2:
        raise ValueError("CNOT требует двух кубитов.")
    _check_qubit(control, n_qubits, "control")
    _check_qubit(target, n_qubits, "target")
    if control == target:
        raise ValueError("Управляющий и целевой кубиты CNOT должны различаться.")
    dim = 2**n_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for index in range(dim):
        bits = [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        new_index = int("".join(map(str, bits)), 2)
        matrix[new_index, index] = 1.0
    return matrix


def validate_operation(op: Operation, n_qubits: int) -> None:
    _check_n_qubits(n_qubits)
    if op.gate not in SUPPORTED_GATES:
        raise ValueError(
            f"Неподдерживаемый гейт {op.gate!r}. Доступны: {', '.join(SUPPORTED_GATES)}."
        )
    _check_qubit(op.target, n_qubits, "target")
    if op.gate == "CNOT":
        if n_qubits < 2:
            raise ValueError("CNOT доступен только в схеме из двух кубитов.")
        if op.control is None:
            raise ValueError("Для CNOT необходимо указать управляющий кубит.")
        _check_qubit(op.control, n_qubits, "control")
        if op.control == op.target:
            raise ValueError("Управляющий и целевой кубиты CNOT должны различаться.")
    elif op.control is not None:
        raise ValueError(f"Гейт {op.gate} не использует управляющий кубит.")
    if op.gate == "RY":
        if op.theta_deg is None or not math.isfinite(op.theta_deg):
            raise ValueError("Для RY необходимо указать конечный угол в градусах.")
    elif op.theta_deg is not None:
        raise ValueError(f"Гейт {op.gate} не использует угол.")


def operation_matrix(op: Operation, n_qubits: int) -> np.ndarray:
    validate_operation(op, n_qubits)
    if op.gate == "CNOT":
        assert op.control is not None
        return cnot_matrix(op.control, op.target, n_qubits)
    return expand_single_qubit(single_qubit_matrix(op.gate, op.theta_deg), op.target, n_qubits)


def apply_operation(state: np.ndarray, op: Operation, n_qubits: int) -> np.ndarray:
    """Применяет одну операцию и проверяет нормировку результата."""
    validate_state(state, n_qubits)
    new_state = operation_matrix(op, n_qubits) @ np.asarray(state, dtype=complex)
    validate_state(new_state, n_qubits)
    return new_state


def apply_gate(
    state: np.ndarray,
    gate: str,
    target: int,
    n_qubits: int,
    control: int | None = None,
    theta_deg: float | None = None,
) -> np.ndarray:
    return apply_operation(
        state, Operation(gate.upper(), target, control, theta_deg), n_qubits
    )


def validate_circuit(operations: Sequence[Operation], n_qubits: int) -> None:
    _check_n_qubits(n_qubits)
    if len(operations) > MAX_OPERATIONS:
        raise ValueError(f"Схема не может содержать более {MAX_OPERATIONS} операций.")
    for position, op in enumerate(operations, start=1):
        try:
            validate_operation(op, n_qubits)
        except ValueError as exc:
            raise ValueError(f"Операция №{position}: {exc}") from exc


def run_circuit_with_history(
    operations: Sequence[Operation], n_qubits: int
) -> list[np.ndarray]:
    """Возвращает список состояний: начальное и после каждой операции."""
    validate_circuit(operations, n_qubits)
    states = [initial_state(n_qubits)]
    for op in operations:
        states.append(apply_operation(states[-1], op, n_qubits))
    return states


def run_circuit(operations: Sequence[Operation], n_qubits: int) -> np.ndarray:
    return run_circuit_with_history(operations, n_qubits)[-1]


def probabilities(state: np.ndarray) -> np.ndarray:
    """Вероятности исходов измерения: |α_i|²."""
    return np.abs(np.asarray(state, dtype=complex)) ** 2


def _sampling_distribution(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("Распределение вероятностей должно быть непустым вектором.")
    if np.any(~np.isfinite(p)) or np.any(p < -NORM_TOLERANCE):
        raise ValueError("Вероятности должны быть неотрицательными числами.")
    total = float(p.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Сумма вероятностей равна {total:.9f}, а не 1.")
    # Значения на уровне погрешности округления считаются нулевыми исходами.
    p = np.where(p < 1e-15, 0.0, p)
    return p / p.sum()


def _check_shots_and_seed(shots: int, seed: int | None) -> None:
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise ValueError("Число измерений должно быть целым.")
    if not MIN_SHOTS <= shots <= MAX_SHOTS:
        raise ValueError(f"Число измерений должно быть от {MIN_SHOTS} до {MAX_SHOTS}.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("Seed должен быть целым числом.")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"Seed должен быть в диапазоне 0…{MAX_SEED}.")


def sample_outcomes(probs: np.ndarray, shots: int, seed: int | None) -> np.ndarray:
    """Индексы исходов для `shots` независимых измерений свежеприготовленных копий."""
    _check_shots_and_seed(shots, seed)
    p = _sampling_distribution(probs)
    rng = np.random.default_rng(seed)
    return rng.choice(p.size, size=int(shots), p=p)


def counts_from_outcomes(outcomes: np.ndarray, n_outcomes: int) -> np.ndarray:
    return np.bincount(np.asarray(outcomes, dtype=int), minlength=n_outcomes)


def sample_counts(probs: np.ndarray, shots: int, seed: int | None) -> np.ndarray:
    """Число появлений каждого базисного исхода."""
    outcomes = sample_outcomes(probs, shots, seed)
    return counts_from_outcomes(outcomes, len(probs))


def total_variation_distance(observed: Iterable[float], theoretical: Iterable[float]) -> float:
    """TVD = 0.5 · Σ |f_i − p_i|."""
    f = np.asarray(list(observed), dtype=float)
    p = np.asarray(list(theoretical), dtype=float)
    if f.shape != p.shape:
        raise ValueError("Распределения должны иметь одинаковую длину.")
    return float(0.5 * np.sum(np.abs(f - p)))


def bloch_vector(state: np.ndarray) -> tuple[float, float, float]:
    """Координаты сферы Блоха для нормированного однокубитного чистого состояния."""
    validate_state(state, 1)
    a, b = complex(state[0]), complex(state[1])
    cross = np.conj(a) * b
    return (
        float(2.0 * cross.real),
        float(2.0 * cross.imag),
        float(abs(a) ** 2 - abs(b) ** 2),
    )
