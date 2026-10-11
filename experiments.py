"""Учебные эксперименты. Все результаты вычисляются тем же движком, что и в лаборатории."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantum import (
    Operation,
    basis_labels,
    counts_from_outcomes,
    probabilities,
    run_circuit,
    sample_outcomes,
    total_variation_distance,
)


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    n_qubits: int
    operations: tuple[Operation, ...]
    goal: str
    expectation: str
    notes: tuple[str, ...] = field(default_factory=tuple)


BIT_FLIP = Preset(
    key="bit_flip",
    title="Эксперимент 1 — Инверсия бита",
    n_qubits=1,
    operations=(Operation("X", 0),),
    goal="Начальное состояние |0⟩, применяется гейт X.",
    expectation="Ожидаемый теоретический результат: |1⟩ с вероятностью 1.",
)

SUPERPOSITION = Preset(
    key="superposition",
    title="Эксперимент 2 — Суперпозиция",
    n_qubits=1,
    operations=(Operation("H", 0),),
    goal="Начальное состояние |0⟩, применяется гейт Адамара H.",
    expectation="Ожидаемые вероятности: 0,5 для |0⟩ и 0,5 для |1⟩.",
    notes=(
        "Состояние (|0⟩ + |1⟩)/√2 — это не «скрытый» выбор между 0 и 1. "
        "Обе амплитуды существуют одновременно и могут интерферировать, "
        "что показывает следующий эксперимент.",
    ),
)

INTERFERENCE_HH = Preset(
    key="interference_hh",
    title="H → H",
    n_qubits=1,
    operations=(Operation("H", 0), Operation("H", 0)),
    goal="Два гейта Адамара подряд.",
    expectation="Амплитуды |1⟩ гасят друг друга, результат — |0⟩ с вероятностью 1.",
)

INTERFERENCE_HZH = Preset(
    key="interference_hzh",
    title="H → Z → H",
    n_qubits=1,
    operations=(Operation("H", 0), Operation("Z", 0), Operation("H", 0)),
    goal="Между гейтами Адамара добавлен фазовый гейт Z.",
    expectation="Гасятся амплитуды |0⟩, результат — |1⟩ с вероятностью 1.",
)

BELL = Preset(
    key="bell",
    title="Эксперимент 4 — Состояние Белла",
    n_qubits=2,
    operations=(Operation("H", 0), Operation("CNOT", target=1, control=0)),
    goal="Начальное состояние |00⟩: H на q0, затем CNOT (управляющий q0, целевой q1).",
    expectation="Ожидаемое конечное состояние: (|00⟩ + |11⟩)/√2.",
    notes=(
        "Совместные результаты измерения коррелированы: выпадают только 00 и 11.",
        "Одной корреляции недостаточно, чтобы доказать запутанность неизвестной "
        "физической системы: похожую статистику в одном базисе даёт и классическая смесь.",
        "Здесь же состояние известно из расчёта: (|00⟩ + |11⟩)/√2 нельзя записать как "
        "произведение состояний отдельных кубитов, поэтому оно запутано.",
        "Запутанность не позволяет передавать сообщения быстрее света: "
        "результаты на каждой стороне по отдельности случайны (0 и 1 с вероятностью 0,5).",
    ),
)

PRESETS: tuple[Preset, ...] = (BIT_FLIP, SUPERPOSITION, INTERFERENCE_HH, INTERFERENCE_HZH, BELL)

INTERFERENCE_EXPLANATION = (
    "После первого H состояние равно (|0⟩ + |1⟩)/√2. Гейт Z меняет знак амплитуды |1⟩: "
    "(|0⟩ − |1⟩)/√2. Вероятности при этом не меняются (по 0,5), меняется только фаза. "
    "Второй H складывает амплитуды: без Z вклады в |1⟩ взаимно уничтожаются и остаётся |0⟩, "
    "с Z взаимно уничтожаются вклады в |0⟩ и остаётся |1⟩. Так относительная фаза, "
    "невидимая в вероятностях одного измерения, определяет результат после интерференции."
)


def preset_by_key(key: str) -> Preset:
    for preset in PRESETS:
        if preset.key == key:
            return preset
    raise KeyError(f"Неизвестный эксперимент: {key}")


def run_preset(preset: Preset) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (амплитуды, вероятности) конечного состояния."""
    state = run_circuit(preset.operations, preset.n_qubits)
    return state, probabilities(state)


# Исследование «Число измерений и отклонение частот».
CONVERGENCE_SHOTS: tuple[int, ...] = (10, 100, 1_000, 10_000)
CONVERGENCE_SEEDS: tuple[int, ...] = tuple(range(1, 31))
CONVERGENCE_METHOD = (
    "Схема: H на q0 (теоретически 0,5 / 0,5). Для каждого seed из списка 1, 2, …, 30 "
    "генератор NumPy (default_rng(seed)) порождает одну серию из 10 000 независимых измерений. "
    "Частоты считаются по первым 10, 100, 1 000 и 10 000 исходам этой серии, "
    "т.е. серия наблюдается по мере накопления измерений. Для каждой точки вычисляется "
    "расстояние полной вариации TVD = 0,5 · Σ|fᵢ − pᵢ|, затем — среднее по 30 сериям."
)


@dataclass(frozen=True)
class ConvergenceResult:
    runs: list[dict[str, float | int]]
    summary: list[dict[str, float | int]]
    non_monotonic_series: int


def run_convergence_experiment(
    shots_list: tuple[int, ...] = CONVERGENCE_SHOTS,
    seeds: tuple[int, ...] = CONVERGENCE_SEEDS,
) -> ConvergenceResult:
    if not shots_list or not seeds:
        raise ValueError("Нужны непустые списки чисел измерений и seed.")
    if list(shots_list) != sorted(set(shots_list)):
        raise ValueError("Числа измерений должны быть различными и возрастать.")
    n_qubits = SUPERPOSITION.n_qubits
    probs = probabilities(run_circuit(SUPERPOSITION.operations, n_qubits))
    n_outcomes = len(probs)
    labels = basis_labels(n_qubits)
    max_shots = shots_list[-1]

    runs: list[dict[str, float | int]] = []
    non_monotonic = 0
    for seed in seeds:
        outcomes = sample_outcomes(probs, max_shots, seed)
        series: list[float] = []
        for shots in shots_list:
            counts = counts_from_outcomes(outcomes[:shots], n_outcomes)
            freqs = counts / shots
            tvd = total_variation_distance(freqs, probs)
            series.append(tvd)
            row: dict[str, float | int] = {"seed": seed, "shots": shots}
            for label, count in zip(labels, counts):
                row[f"count {label}"] = int(count)
            row["tvd"] = tvd
            runs.append(row)
        if any(later > earlier for earlier, later in zip(series, series[1:])):
            non_monotonic += 1

    summary: list[dict[str, float | int]] = []
    for shots in shots_list:
        values = np.array([r["tvd"] for r in runs if r["shots"] == shots], dtype=float)
        summary.append(
            {
                "shots": shots,
                "replications": int(values.size),
                "mean_tvd": float(values.mean()),
                "std_tvd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min_tvd": float(values.min()),
                "max_tvd": float(values.max()),
                "normal_approx": float(1.0 / np.sqrt(2.0 * np.pi * shots)),
            }
        )
    return ConvergenceResult(runs=runs, summary=summary, non_monotonic_series=non_monotonic)
