"""Экспорт: конфигурация схемы (JSON), результаты (CSV, UTF-8 с BOM), HTML-отчёт."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any, Sequence

import numpy as np

from circuit_view import circuit_svg, describe_operation, operation_label
from quantum import (
    BASIS_CONVENTION,
    MAX_OPERATIONS,
    MODEL_VERSION,
    Operation,
    basis_labels,
    validate_circuit,
)

LIMITATIONS: tuple[str, ...] = (
    "Вычисления выполняются на классическом компьютере; подключения к квантовому "
    "оборудованию нет.",
    "Поддерживаются только один или два кубита и гейты X, H, Z, RY, CNOT.",
    "Шум, декогеренция и аппаратные ошибки не моделируются.",
    "Измерения моделируются псевдослучайным генератором NumPy по рассчитанным вероятностям.",
    "Образовательная эффективность требует отдельного исследования с участием учащихся.",
)

CONFIG_FORMAT = "quantumlab-circuit"


@dataclass(frozen=True)
class Measurement:
    shots: int
    seed: int
    counts: np.ndarray

    @property
    def frequencies(self) -> np.ndarray:
        return self.counts / self.shots


def circuit_config(operations: Sequence[Operation], n_qubits: int) -> dict[str, Any]:
    validate_circuit(operations, n_qubits)
    return {
        "format": CONFIG_FORMAT,
        "model_version": MODEL_VERSION,
        "n_qubits": n_qubits,
        "basis_convention": BASIS_CONVENTION,
        "basis_order": basis_labels(n_qubits),
        "angle_unit": "degrees",
        "operations": [op.to_dict() for op in operations],
    }


def config_to_json(operations: Sequence[Operation], n_qubits: int) -> str:
    return json.dumps(circuit_config(operations, n_qubits), ensure_ascii=False, indent=2)


def config_from_json(text: str) -> tuple[int, list[Operation]]:
    """Разбирает и проверяет конфигурацию. Бросает ValueError при ошибках."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Файл не является корректным JSON: {exc.msg}.") from exc
    if not isinstance(data, dict) or data.get("format") != CONFIG_FORMAT:
        raise ValueError("Это не файл конфигурации QuantumLab.")
    n_qubits = data.get("n_qubits")
    raw_ops = data.get("operations")
    if not isinstance(raw_ops, list):
        raise ValueError("Поле operations должно быть списком.")
    if len(raw_ops) > MAX_OPERATIONS:
        raise ValueError(f"Схема не может содержать более {MAX_OPERATIONS} операций.")
    operations = [Operation.from_dict(item) for item in raw_ops]
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise ValueError("Поле n_qubits должно быть целым числом.")
    validate_circuit(operations, n_qubits)
    return n_qubits, operations


def results_rows(
    state: np.ndarray, n_qubits: int, measurement: Measurement | None
) -> list[dict[str, Any]]:
    probs = np.abs(state) ** 2
    rows = []
    for i, label in enumerate(basis_labels(n_qubits)):
        row: dict[str, Any] = {
            "basis": label,
            "re": float(state[i].real),
            "im": float(state[i].imag),
            "probability": float(probs[i]),
        }
        if measurement is not None:
            row["count"] = int(measurement.counts[i])
            row["frequency"] = float(measurement.frequencies[i])
        rows.append(row)
    return rows


def _csv_bytes(header: list[str], rows: list[list[Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def measurements_csv(
    state: np.ndarray, n_qubits: int, measurement: Measurement
) -> bytes:
    header = [
        "Базисное состояние",
        "Re(амплитуды)",
        "Im(амплитуды)",
        "Теоретическая вероятность",
        "Число исходов (симуляция)",
        "Наблюдаемая частота (симуляция)",
        "Число измерений",
        "Seed",
        "Число кубитов",
        "Версия модели",
    ]
    rows = [
        [
            r["basis"],
            repr(r["re"]),
            repr(r["im"]),
            repr(r["probability"]),
            r["count"],
            repr(r["frequency"]),
            measurement.shots,
            measurement.seed,
            n_qubits,
            MODEL_VERSION,
        ]
        for r in results_rows(state, n_qubits, measurement)
    ]
    return _csv_bytes(header, rows)


def convergence_csv(runs: list[dict[str, Any]]) -> bytes:
    if not runs:
        return _csv_bytes(["seed", "shots", "tvd"], [])
    header = list(runs[0].keys()) + ["model_version"]
    rows = [[*(r[k] for k in runs[0].keys()), MODEL_VERSION] for r in runs]
    return _csv_bytes(header, rows)


def _fmt(value: float, digits: int = 6) -> str:
    rounded = round(float(value), digits) + 0.0
    return f"{rounded:.{digits}f}"


def html_report(
    operations: Sequence[Operation],
    n_qubits: int,
    state: np.ndarray,
    measurement: Measurement | None,
    generated_at: datetime | None = None,
) -> str:
    """Самостоятельный HTML-отчёт. Весь текст экранируется."""
    validate_circuit(operations, n_qubits)
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    rows = results_rows(state, n_qubits, measurement)
    norm = float(np.sqrt(np.sum(np.abs(state) ** 2)))

    def cell(value: Any) -> str:
        return f"<td>{escape(str(value))}</td>"

    ops_html = "".join(
        f"<li><code>{escape(operation_label(op))}</code> — {escape(describe_operation(op))}"
        f"<br><small>{escape(json.dumps(op.to_dict(), ensure_ascii=False))}</small></li>"
        for op in operations
    ) or "<li>Операций нет: конечное состояние совпадает с начальным.</li>"

    amp_rows = "".join(
        "<tr>"
        + cell(r["basis"])
        + cell(_fmt(r["re"]))
        + cell(_fmt(r["im"]))
        + cell(_fmt(r["probability"]))
        + "</tr>"
        for r in rows
    )

    if measurement is not None:
        meas_rows = "".join(
            "<tr>"
            + cell(r["basis"])
            + cell(r["count"])
            + cell(_fmt(r["frequency"], 4))
            + cell(_fmt(r["probability"], 4))
            + "</tr>"
            for r in rows
        )
        measurement_html = (
            f"<p>Число измерений: <b>{measurement.shots}</b>, seed: "
            f"<b>{measurement.seed}</b>. Каждое измерение выполняется над заново "
            "приготовленной копией конечного состояния.</p>"
            "<table><thead><tr><th>Состояние</th><th>Число исходов (симуляция)</th>"
            "<th>Частота (симуляция)</th><th>Теоретическая вероятность</th></tr></thead>"
            f"<tbody>{meas_rows}</tbody></table>"
        )
    else:
        measurement_html = "<p>Моделирование измерений не выполнялось.</p>"

    limitations = "".join(f"<li>{escape(item)}</li>" for item in LIMITATIONS)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>QuantumLab — отчёт</title>
<style>
body {{ background:#F7F8FA; color:#202733; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin:0; }}
main {{ max-width:1000px; margin:0 auto; padding:32px 24px; }}
section {{ background:#FFFFFF; border:1px solid #E4E7EC; border-radius:6px; padding:16px 20px; margin-bottom:16px; }}
h1 {{ margin:0 0 4px; font-size:26px; }}
h2 {{ font-size:17px; margin:0 0 12px; }}
.muted, small {{ color:#667085; }}
code, td {{ font-family: Consolas, "Courier New", monospace; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ border:1px solid #E4E7EC; padding:6px 10px; text-align:left; }}
th {{ background:#F7F8FA; font-weight:600; }}
.diagram {{ overflow-x:auto; }}
</style>
</head>
<body><main>
<h1>QuantumLab</h1>
<p class="muted">Виртуальная лаборатория квантовых вычислений — отчёт от {escape(stamp)}.
Версия модели {escape(MODEL_VERSION)}. Расчёт выполнен на классическом компьютере.</p>

<section><h2>Параметры схемы</h2>
<p>Число кубитов: <b>{n_qubits}</b>. Число операций: <b>{len(operations)}</b>.
Проверка нормировки: ‖ψ‖ = {escape(_fmt(norm, 9))}.</p>
<p class="muted">{escape(BASIS_CONVENTION)}</p>
<div class="diagram">{circuit_svg(operations, n_qubits)}</div>
<ol>{ops_html}</ol>
</section>

<section><h2>Конечное состояние и теоретические вероятности</h2>
<p>Амплитуды описывают квантовое состояние. Вероятности результатов измерения равны
квадратам модулей амплитуд.</p>
<table><thead><tr><th>Базисное состояние</th><th>Re(амплитуды)</th><th>Im(амплитуды)</th>
<th>Вероятность</th></tr></thead><tbody>{amp_rows}</tbody></table>
</section>

<section><h2>Моделирование измерений</h2>{measurement_html}</section>

<section><h2>Ограничения</h2><ul>{limitations}</ul></section>
</main></body></html>
"""
