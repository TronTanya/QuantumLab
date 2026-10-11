"""Детерминированная SVG-схема и текстовые описания операций."""

from __future__ import annotations

from html import escape
from typing import Sequence

import numpy as np

from quantum import Operation, validate_circuit

TEXT = "#202733"
MUTED = "#667085"
BORDER = "#E4E7EC"
ACCENT = "#315EFB"
SURFACE = "#FFFFFF"
MONO = "'JetBrains Mono', 'Fira Code', Consolas, 'Courier New', monospace"

LEFT_MARGIN = 56
COLUMN_WIDTH = 56
TOP_MARGIN = 34
WIRE_GAP = 56
BOX = 40


def format_angle(theta_deg: float) -> str:
    return f"{theta_deg:g}°"


def operation_label(op: Operation) -> str:
    """Короткая моноширинная запись операции, например RY(90°) q0."""
    if op.gate == "CNOT":
        return f"CNOT q{op.control}→q{op.target}"
    if op.gate == "RY" and op.theta_deg is not None:
        return f"RY({format_angle(op.theta_deg)}) q{op.target}"
    return f"{op.gate} q{op.target}"


def describe_operation(op: Operation) -> str:
    """Пояснение операции на русском языке."""
    q = f"q{op.target}"
    if op.gate == "X":
        return (
            f"Гейт X на {q}: меняет местами амплитуды компонент с {q} = 0 и {q} = 1 "
            "(квантовый аналог логического НЕ)."
        )
    if op.gate == "H":
        return (
            f"Гейт Адамара H на {q}: переводит |0⟩ в (|0⟩ + |1⟩)/√2, "
            "а |1⟩ — в (|0⟩ − |1⟩)/√2."
        )
    if op.gate == "Z":
        return (
            f"Гейт Z на {q}: меняет знак амплитуд компонент с {q} = 1 "
            "(фазовый сдвиг на π, вероятности сразу не меняются)."
        )
    if op.gate == "RY":
        angle = format_angle(op.theta_deg or 0.0)
        return (
            f"Поворот RY({angle}) на {q}: вращение вокруг оси Y сферы Блоха; "
            f"из |0⟩ получается cos(θ/2)|0⟩ + sin(θ/2)|1⟩."
        )
    if op.gate == "CNOT":
        return (
            f"CNOT, управляющий q{op.control}, целевой q{op.target}: инвертирует "
            f"q{op.target} в тех компонентах состояния, где q{op.control} = 1."
        )
    return f"Операция {escape(op.gate)}."


def _format_coefficient(value: complex, digits: int) -> str:
    re = round(value.real, digits) + 0.0
    im = round(value.imag, digits) + 0.0
    if im == 0:
        return f"{re:.{digits}f}"
    if re == 0:
        return f"{im:.{digits}f}i"
    sign = "+" if im > 0 else "−"
    return f"({re:.{digits}f} {sign} {abs(im):.{digits}f}i)"


def format_state(state: np.ndarray, n_qubits: int, digits: int = 4) -> str:
    """Запись состояния в нотации Дирака, например 0.7071|00⟩ + 0.7071|11⟩."""
    terms: list[str] = []
    for index, amplitude in enumerate(np.asarray(state, dtype=complex)):
        if abs(amplitude) < 0.5 * 10**-digits:
            continue
        coefficient = _format_coefficient(complex(amplitude), digits)
        label = f"|{index:0{n_qubits}b}⟩"
        if terms and coefficient.startswith("-"):
            terms.append(f"− {coefficient[1:]}{label}")
        elif terms:
            terms.append(f"+ {coefficient}{label}")
        else:
            terms.append(f"{coefficient}{label}")
    return " ".join(terms) if terms else "0"


def _wire_y(qubit: int) -> int:
    return TOP_MARGIN + qubit * WIRE_GAP


def _gate_box(cx: int, cy: int, op: Operation) -> str:
    x, y = cx - BOX // 2, cy - BOX // 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{BOX}" height="{BOX}" rx="4" '
        f'fill="{SURFACE}" stroke="{ACCENT}" stroke-width="1.5"/>'
    ]
    if op.gate == "RY":
        angle = escape(format_angle(op.theta_deg or 0.0))
        parts.append(
            f'<text x="{cx}" y="{cy - 3}" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="{TEXT}">RY</text>'
            f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="9.5" '
            f'fill="{MUTED}">{angle}</text>'
        )
    else:
        parts.append(
            f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="{TEXT}">{escape(op.gate)}</text>'
        )
    return "".join(parts)


def _cnot(cx: int, op: Operation) -> str:
    assert op.control is not None
    cy_c, cy_t = _wire_y(op.control), _wire_y(op.target)
    radius = 12
    direction = 1 if cy_t > cy_c else -1
    return (
        f'<line x1="{cx}" y1="{cy_c}" x2="{cx}" y2="{cy_t + direction * radius}" '
        f'stroke="{ACCENT}" stroke-width="1.5"/>'
        f'<circle cx="{cx}" cy="{cy_c}" r="5" fill="{ACCENT}"/>'
        f'<circle cx="{cx}" cy="{cy_t}" r="{radius}" fill="{SURFACE}" '
        f'stroke="{ACCENT}" stroke-width="1.5"/>'
        f'<line x1="{cx - radius}" y1="{cy_t}" x2="{cx + radius}" y2="{cy_t}" '
        f'stroke="{ACCENT}" stroke-width="1.5"/>'
        f'<line x1="{cx}" y1="{cy_t - radius}" x2="{cx}" y2="{cy_t + radius}" '
        f'stroke="{ACCENT}" stroke-width="1.5"/>'
    )


def circuit_svg(operations: Sequence[Operation], n_qubits: int) -> str:
    """SVG-диаграмма: одна колонка на операцию, слева направо в порядке применения."""
    validate_circuit(operations, n_qubits)
    columns = max(len(operations), 1)
    width = LEFT_MARGIN + columns * COLUMN_WIDTH + 24
    height = TOP_MARGIN + (n_qubits - 1) * WIRE_GAP + 34
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{MONO}" role="img" '
        f'aria-label="Диаграмма квантовой схемы">'
    ]
    for q in range(n_qubits):
        y = _wire_y(q)
        parts.append(
            f'<text x="8" y="{y + 5}" font-size="13" fill="{TEXT}">q{q}</text>'
            f'<text x="28" y="{y + 5}" font-size="11" fill="{MUTED}">|0⟩</text>'
            f'<line x1="{LEFT_MARGIN - 4}" y1="{y}" x2="{width - 12}" y2="{y}" '
            f'stroke="{MUTED}" stroke-width="1"/>'
        )
    for index, op in enumerate(operations):
        cx = LEFT_MARGIN + index * COLUMN_WIDTH + COLUMN_WIDTH // 2
        parts.append(
            f'<text x="{cx}" y="11" text-anchor="middle" font-size="10" '
            f'fill="{MUTED}">{index + 1}</text>'
        )
        if op.gate == "CNOT":
            parts.append(_cnot(cx, op))
        else:
            parts.append(_gate_box(cx, _wire_y(op.target), op))
    if not operations:
        parts.append(
            f'<text x="{LEFT_MARGIN + 6}" y="{height - 6}" font-size="11" '
            f'fill="{MUTED}">операций нет</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def circuit_html(operations: Sequence[Operation], n_qubits: int) -> str:
    """SVG в рамке с горизонтальной прокруткой для длинных схем."""
    svg = circuit_svg(operations, n_qubits)
    return (
        f'<div style="overflow-x:auto;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:6px;padding:10px 8px;">{svg}</div>'
    )
