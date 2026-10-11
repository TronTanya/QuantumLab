"""QuantumLab — интерфейс Streamlit. Вычисления находятся в quantum.py и experiments.py."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from circuit_view import circuit_html, describe_operation, format_state, operation_label
from experiments import (
    BELL,
    BIT_FLIP,
    CONVERGENCE_METHOD,
    CONVERGENCE_SEEDS,
    CONVERGENCE_SHOTS,
    INTERFERENCE_EXPLANATION,
    INTERFERENCE_HH,
    INTERFERENCE_HZH,
    SUPERPOSITION,
    Preset,
    preset_by_key,
    run_convergence_experiment,
)
from quantum import (
    BASIS_CONVENTION,
    MAX_OPERATIONS,
    MODEL_VERSION,
    NORM_TOLERANCE,
    Operation,
    basis_labels,
    bloch_vector,
    probabilities,
    run_circuit_with_history,
    sample_counts,
    state_norm,
    validate_operation,
)
from reporting import (
    LIMITATIONS,
    Measurement,
    config_from_json,
    config_to_json,
    convergence_csv,
    html_report,
    measurements_csv,
)

TEXT = "#202733"
MUTED = "#667085"
BORDER = "#E4E7EC"
ACCENT = "#315EFB"
SIM_COLOR = "#98A2B3"
MONO = "Consolas, 'Courier New', monospace"

GATE_NAMES = {
    "X": "X — инверсия",
    "H": "H — Адамар",
    "Z": "Z — фаза",
    "RY": "RY(θ) — поворот",
    "CNOT": "CNOT — управляемое НЕ",
}

st.set_page_config(page_title="QuantumLab", layout="wide")
# data-testid — стабильный атрибут Streamlit; ограничиваем ширину контента ~1100px.
st.markdown(
    "<style>[data-testid='stMainBlockContainer']{max-width:1100px;}</style>",
    unsafe_allow_html=True,
)


# ---------- Состояние сессии ----------

def _init_state() -> None:
    defaults = {
        "n_qubits": 1,
        "ops": [],
        "result": None,
        "measurement": None,
        "message": None,
        "gate_choice": "H",
        "target_q": 0,
        "cnot_control": 0,
        "theta_deg": 90.0,
        "shots": 1000,
        "seed": 42,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _invalidate() -> None:
    st.session_state.result = None
    st.session_state.measurement = None


def _set_message(kind: str, text: str) -> None:
    st.session_state.message = (kind, text)


def _on_qubits_change() -> None:
    n = st.session_state.n_qubits
    had_ops = bool(st.session_state.ops)
    st.session_state.ops = []
    _invalidate()
    if n == 1:
        if st.session_state.gate_choice == "CNOT":
            st.session_state.gate_choice = "H"
        st.session_state.target_q = 0
        st.session_state.cnot_control = 0
    if had_ops:
        _set_message("info", f"Число кубитов изменено на {n}: схема очищена.")


def _add_gate() -> None:
    ops: list[Operation] = st.session_state.ops
    n = st.session_state.n_qubits
    if len(ops) >= MAX_OPERATIONS:
        _set_message("error", f"Достигнут предел в {MAX_OPERATIONS} операций.")
        return
    gate = st.session_state.gate_choice
    if gate == "CNOT":
        control = int(st.session_state.cnot_control)
        op = Operation("CNOT", target=1 - control, control=control)
    elif gate == "RY":
        op = Operation("RY", int(st.session_state.target_q), theta_deg=float(st.session_state.theta_deg))
    else:
        op = Operation(gate, int(st.session_state.target_q))
    try:
        validate_operation(op, n)
    except ValueError as exc:
        _set_message("error", str(exc))
        return
    st.session_state.ops = [*ops, op]
    _invalidate()


def _remove_last() -> None:
    if st.session_state.ops:
        st.session_state.ops = st.session_state.ops[:-1]
        _invalidate()


def _clear() -> None:
    st.session_state.ops = []
    _invalidate()


def _calculate() -> None:
    n = st.session_state.n_qubits
    ops = list(st.session_state.ops)
    try:
        states = run_circuit_with_history(ops, n)
    except ValueError as exc:
        _set_message("error", f"Расчёт невозможен: {exc}")
        st.session_state.result = None
        return
    st.session_state.result = {"n": n, "ops": ops, "states": states}
    st.session_state.measurement = None


def _simulate() -> None:
    result = st.session_state.result
    if result is None:
        return
    shots, seed = int(st.session_state.shots), int(st.session_state.seed)
    try:
        counts = sample_counts(probabilities(result["states"][-1]), shots, seed)
    except ValueError as exc:
        _set_message("error", str(exc))
        return
    st.session_state.measurement = Measurement(shots=shots, seed=seed, counts=counts)


def _load_preset(key: str) -> None:
    preset = preset_by_key(key)
    _load_circuit(preset.n_qubits, list(preset.operations))
    _set_message("success", f"Схема «{preset.title}» загружена во вкладку «Лаборатория».")


def _load_circuit(n: int, ops: list[Operation]) -> None:
    st.session_state.n_qubits = n
    st.session_state.ops = ops
    if n == 1:
        st.session_state.target_q = 0
        if st.session_state.gate_choice == "CNOT":
            st.session_state.gate_choice = "H"
    _calculate()


def _import_config() -> None:
    uploaded = st.session_state.get("config_file")
    if uploaded is None:
        _set_message("error", "Сначала выберите файл JSON.")
        return
    try:
        n, ops = config_from_json(uploaded.getvalue().decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        _set_message("error", f"Не удалось загрузить конфигурацию: {exc}")
        return
    _load_circuit(n, ops)
    _set_message("success", f"Конфигурация загружена: {n} кубит(а), операций: {len(ops)}.")


# ---------- Таблицы и графики ----------

def _round(x: float, digits: int = 6) -> float:
    return round(float(x), digits) + 0.0


def amplitude_table(state: np.ndarray, n: int) -> pd.DataFrame:
    probs = probabilities(state)
    return pd.DataFrame(
        {
            "Базисное состояние": basis_labels(n),
            "Re(амплитуды)": [_round(a.real) for a in state],
            "Im(амплитуды)": [_round(a.imag) for a in state],
            "Вероятность": [_round(p) for p in probs],
        }
    )


AMPLITUDE_COLUMNS = {
    "Re(амплитуды)": st.column_config.NumberColumn(format="%.6f"),
    "Im(амплитуды)": st.column_config.NumberColumn(format="%.6f"),
    "Вероятность": st.column_config.NumberColumn(format="%.6f"),
}


def measurement_table(state: np.ndarray, n: int, m: Measurement) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Базисное состояние": basis_labels(n),
            "Число исходов (симуляция)": m.counts.astype(int),
            "Частота (симуляция)": [_round(f, 4) for f in m.frequencies],
            "Теоретическая вероятность": [_round(p, 4) for p in probabilities(state)],
        }
    )


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    return fig


def comparison_chart(state: np.ndarray, n: int, m: Measurement) -> go.Figure:
    labels = basis_labels(n)
    fig = go.Figure(
        [
            go.Bar(
                name="Теоретическая вероятность",
                x=labels,
                y=probabilities(state),
                marker_color=ACCENT,
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ),
            go.Bar(
                name=f"Частота — результат симуляции ({m.shots} изм., seed {m.seed})",
                x=labels,
                y=m.frequencies,
                marker_color=SIM_COLOR,
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="Вероятности и результаты измерений", font=dict(size=16)),
        yaxis=dict(title="Доля", range=[0, 1.05], gridcolor=BORDER),
        xaxis=dict(title="Базисное состояние |q0 q1⟩" if n == 2 else "Базисное состояние",
                   tickfont=dict(family=MONO, size=14)),
    )
    return _base_layout(fig, 460)


def bloch_figure(state: np.ndarray) -> go.Figure:
    x, y, z = bloch_vector(state)
    t = np.linspace(0, 2 * np.pi, 121)
    fig = go.Figure()
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    fig.add_trace(
        go.Surface(
            x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
            opacity=0.12, showscale=False, hoverinfo="skip",
            colorscale=[[0, "#C7D2FE"], [1, "#C7D2FE"]],
        )
    )
    for cx, cy, cz in (
        (np.cos(t), np.sin(t), 0 * t),
        (np.cos(t), 0 * t, np.sin(t)),
        (0 * t, np.cos(t), np.sin(t)),
    ):
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                                   line=dict(color="#B8C0CC", width=2), hoverinfo="skip"))
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        a = np.array(axis) * 1.2
        fig.add_trace(go.Scatter3d(x=[-a[0], a[0]], y=[-a[1], a[1]], z=[-a[2], a[2]],
                                   mode="lines", line=dict(color=MUTED, width=2),
                                   hoverinfo="skip"))
    fig.add_trace(
        go.Scatter3d(
            x=[1.38, -1.38, 0, 0, 0, 0], y=[0, 0, 1.38, -1.38, 0, 0], z=[0, 0, 0, 0, 1.3, -1.4],
            mode="text",
            text=["x  |+⟩", "|−⟩", "y  |+i⟩", "|−i⟩", "z  |0⟩", "|1⟩"],
            textfont=dict(family=MONO, size=13, color=TEXT), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines",
                     line=dict(color=ACCENT, width=7),
                     hovertemplate=f"x={x:.3f}, y={y:.3f}, z={z:.3f}<extra></extra>")
    )
    fig.add_trace(
        go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute", sizeref=0.18,
                anchor="tip", showscale=False, colorscale=[[0, ACCENT], [1, ACCENT]],
                hoverinfo="skip")
    )
    hidden = dict(visible=False, range=[-1.45, 1.45])
    fig.update_layout(
        showlegend=False,
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="cube",
                   camera=dict(eye=dict(x=1.35, y=1.0, z=0.7))),
        title=dict(text="Сфера Блоха", font=dict(size=16)),
    )
    return _base_layout(fig, 460)


# ---------- Вкладка «Лаборатория» ----------

def render_message() -> None:
    message = st.session_state.message
    if message:
        kind, text = message
        getattr(st, kind)(text)
        st.session_state.message = None


def render_builder() -> None:
    n = st.session_state.n_qubits
    ops: list[Operation] = st.session_state.ops

    with st.container(border=True):
        st.markdown("##### 1. Число кубитов")
        st.radio(
            "Число кубитов",
            options=[1, 2],
            key="n_qubits",
            horizontal=True,
            on_change=_on_qubits_change,
            format_func=lambda k: f"{k} кубит" if k == 1 else f"{k} кубита",
            label_visibility="collapsed",
        )
        start = "|0⟩" if n == 1 else "|00⟩"
        st.caption(
            f"Начальное состояние: `{start}`. При смене числа кубитов схема очищается. "
            + (BASIS_CONVENTION if n == 2 else "")
        )

    with st.container(border=True):
        st.markdown("##### 2. Добавление гейта")
        gates = ["X", "H", "Z", "RY"] + (["CNOT"] if n == 2 else [])
        if st.session_state.gate_choice not in gates:
            st.session_state.gate_choice = "H"
        c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1])
        with c1:
            st.selectbox("Гейт", gates, key="gate_choice", format_func=GATE_NAMES.get)
        gate = st.session_state.gate_choice
        with c2:
            if gate == "CNOT":
                st.selectbox("Управляющий кубит", [0, 1], key="cnot_control",
                             format_func=lambda q: f"q{q}")
            else:
                st.selectbox("Целевой кубит", list(range(n)), key="target_q",
                             format_func=lambda q: f"q{q}")
        with c3:
            if gate == "CNOT":
                st.text_input("Целевой кубит", value=f"q{1 - st.session_state.cnot_control}",
                              disabled=True)
            elif gate == "RY":
                st.number_input("Угол θ, градусы", min_value=-720.0, max_value=720.0,
                                step=15.0, key="theta_deg")
        with c4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("Добавить", on_click=_add_gate, width="stretch",
                      disabled=len(ops) >= MAX_OPERATIONS)
        if gate == "RY":
            st.caption("Угол вводится в градусах и переводится в радианы при расчёте.")

    with st.container(border=True):
        st.markdown(f"##### 3. Схема — операций: {len(ops)} из {MAX_OPERATIONS}")
        st.markdown(circuit_html(ops, n), unsafe_allow_html=True)
        if ops:
            lines = "\n".join(
                f"{i}. `{operation_label(op)}` — {describe_operation(op)}"
                for i, op in enumerate(ops, start=1)
            )
            st.markdown(lines)
        else:
            st.caption("Схема пуста. Добавьте гейты или загрузите эксперимент.")
        b1, b2, b3, _ = st.columns([1, 1, 1, 1.5])
        b1.button("Удалить последнюю", on_click=_remove_last, disabled=not ops,
                  width="stretch")
        b2.button("Очистить схему", on_click=_clear, disabled=not ops,
                  width="stretch")
        b3.button("Рассчитать", on_click=_calculate, type="primary",
                  width="stretch")


def render_results() -> None:
    result = st.session_state.result
    if result is None:
        st.info("Результаты отсутствуют или устарели после изменения схемы. "
                "Нажмите «Рассчитать».")
        return
    n, ops, states = result["n"], result["ops"], result["states"]
    state = states[-1]
    norm = state_norm(state)

    with st.container(border=True):
        st.markdown("##### 4. Конечное состояние")
        m1, m2, m3 = st.columns(3)
        m1.metric("Число кубитов", n)
        m2.metric("Число операций", len(ops))
        m3.metric("Нормировка ‖ψ‖", f"{norm:.9f}",
                  help=f"Допуск: |‖ψ‖ − 1| ≤ {NORM_TOLERANCE:g}")
        if abs(norm - 1) <= NORM_TOLERANCE:
            st.caption(f"Проверка нормировки пройдена: Σ|αᵢ|² = 1 в пределах {NORM_TOLERANCE:g}.")
        else:
            st.error("Состояние не нормировано — проверьте модель.")
        st.markdown(f"Конечное состояние: `{format_state(state, n)}`")
        st.markdown(
            "Амплитуды описывают квантовое состояние. Вероятности результатов измерения "
            "равны квадратам модулей амплитуд."
        )
        st.dataframe(amplitude_table(state, n), hide_index=True, width="stretch",
                     column_config=AMPLITUDE_COLUMNS)
        st.caption(
            "Суперпозиция — это не скрытый классический выбор одного из базисных состояний: "
            "до измерения система описывается всеми амплитудами сразу, включая их знаки (фазы)."
        )
        with st.expander("Пошаговая эволюция состояния"):
            steps = [{"Шаг": 0, "Операция": "начальное состояние",
                      "Состояние": format_state(states[0], n)}]
            for i, (op, s) in enumerate(zip(ops, states[1:]), start=1):
                steps.append({"Шаг": i, "Операция": operation_label(op),
                              "Состояние": format_state(s, n)})
            st.dataframe(pd.DataFrame(steps), hide_index=True, width="stretch")

    if n == 1:
        with st.container(border=True):
            if st.toggle("Показать сферу Блоха", value=False):
                bx, by, bz = bloch_vector(state)
                st.plotly_chart(bloch_figure(state), width="stretch")
                st.caption(
                    f"Координаты: x = {_round(bx, 4)}, y = {_round(by, 4)}, z = {_round(bz, 4)}. "
                    "Для состояния [a, b]: x = 2·Re(a*·b), y = 2·Im(a*·b), z = |a|² − |b|². "
                    "Это изображение применимо к поддерживаемым здесь чистым состояниям одного кубита."
                )
            else:
                st.caption("Сфера Блоха наглядно показывает чистое состояние одного кубита.")
    else:
        st.caption("Сфера Блоха не показывается: двухкубитное состояние (в том числе запутанное) "
                   "нельзя изобразить одной точкой на сфере.")


def render_measurements() -> None:
    result = st.session_state.result
    if result is None:
        return
    n, state = result["n"], result["states"][-1]
    with st.container(border=True):
        st.markdown("##### 5. Моделирование измерений")
        st.markdown(
            "Каждое измерение (shot) выполняется над **заново приготовленной копией** конечного "
            "состояния схемы, поэтому исходы независимы и распределены по теоретическим "
            "вероятностям. Повторное измерение одного и того же уже измеренного (схлопнувшегося) "
            "состояния — другой эксперимент: оно давало бы тот же результат снова."
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.number_input("Число измерений", min_value=10, max_value=10_000, step=10, key="shots")
        c2.number_input("Seed генератора", min_value=0, max_value=2**32 - 1, step=1, key="seed")
        with c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("Смоделировать измерения", on_click=_simulate, width="stretch")

        m: Measurement | None = st.session_state.measurement
        if m is None:
            st.caption("Измерения ещё не моделировались для текущего расчёта.")
            return
        if (m.shots, m.seed) != (st.session_state.shots, st.session_state.seed):
            st.warning("Параметры изменены: показаны результаты для "
                       f"{m.shots} измерений и seed {m.seed}. Нажмите кнопку, чтобы обновить.")
        st.dataframe(measurement_table(state, n, m), hide_index=True, width="stretch")
        st.plotly_chart(comparison_chart(state, n, m), width="stretch")
        tvd = 0.5 * float(np.sum(np.abs(m.frequencies - probabilities(state))))
        st.caption(
            f"Расстояние полной вариации между частотами симуляции и вероятностями: {tvd:.4f}. "
            "Частоты случайны и не обязаны точно совпадать с вероятностями."
        )


def render_export() -> None:
    n, ops = st.session_state.n_qubits, st.session_state.ops
    result, m = st.session_state.result, st.session_state.measurement
    with st.container(border=True):
        st.markdown("##### 6. Экспорт")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Схема (JSON)", config_to_json(ops, n),
                           file_name="quantumlab_circuit.json", mime="application/json",
                           width="stretch")
        if result is not None and m is not None:
            c2.download_button("Результаты измерений (CSV)",
                               measurements_csv(result["states"][-1], n, m),
                               file_name="quantumlab_measurements.csv", mime="text/csv",
                               width="stretch")
        else:
            c2.button("Результаты измерений (CSV)", disabled=True, width="stretch",
                      help="Сначала рассчитайте схему и смоделируйте измерения.")
        if result is not None:
            report = html_report(ops, n, result["states"][-1], m, datetime.now())
            c3.download_button("Отчёт (HTML)", report.encode("utf-8"),
                               file_name="quantumlab_report.html", mime="text/html",
                               width="stretch")
        else:
            c3.button("Отчёт (HTML)", disabled=True, width="stretch",
                      help="Сначала рассчитайте схему.")
        with st.expander("Загрузить конфигурацию схемы из JSON"):
            st.file_uploader("Файл конфигурации QuantumLab", type=["json"], key="config_file")
            st.button("Применить конфигурацию", on_click=_import_config)


def render_lab() -> None:
    render_builder()
    render_results()
    render_measurements()
    render_export()


# ---------- Вкладка «Эксперименты» ----------

def _preset_block(preset: Preset, show_button: bool = True) -> np.ndarray:
    states = run_circuit_with_history(list(preset.operations), preset.n_qubits)
    state = states[-1]
    st.markdown(preset.goal)
    st.markdown(circuit_html(list(preset.operations), preset.n_qubits), unsafe_allow_html=True)
    st.markdown(f"Рассчитанное состояние: `{format_state(state, preset.n_qubits)}`")
    st.dataframe(amplitude_table(state, preset.n_qubits), hide_index=True,
                 width="stretch", column_config=AMPLITUDE_COLUMNS)
    st.caption(preset.expectation)
    if show_button:
        st.button("Открыть в лаборатории", key=f"load_{preset.key}",
                  on_click=_load_preset, args=(preset.key,))
    return state


@st.cache_data(show_spinner=False)
def _convergence() -> tuple[pd.DataFrame, pd.DataFrame, int, bytes]:
    res = run_convergence_experiment()
    return pd.DataFrame(res.runs), pd.DataFrame(res.summary), res.non_monotonic_series, \
        convergence_csv(res.runs)


def render_convergence() -> None:
    with st.container(border=True):
        st.markdown("#### Исследование: число измерений и отклонение частот")
        st.markdown(CONVERGENCE_METHOD)
        runs, summary, non_monotonic, csv_bytes = _convergence()
        table = pd.DataFrame(
            {
                "Число измерений": summary["shots"],
                "Повторений": summary["replications"],
                "Среднее TVD": summary["mean_tvd"].round(4),
                "Ст. отклонение": summary["std_tvd"].round(4),
                "Минимум": summary["min_tvd"].round(4),
                "Максимум": summary["max_tvd"].round(4),
                "Ориентир 1/√(2πN)": summary["normal_approx"].round(4),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

        fig = go.Figure()
        for i, (seed, group) in enumerate(runs.groupby("seed")):
            fig.add_trace(go.Scatter(
                x=group["shots"], y=group["tvd"], mode="lines",
                line=dict(color="#C9CED6", width=1),
                name="Отдельные серии (seed 1…30)", legendgroup="runs", showlegend=i == 0,
                hovertemplate=f"seed {seed}: N=%{{x}}, TVD=%{{y:.4f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(x=summary["shots"], y=summary["mean_tvd"], mode="lines+markers",
                                 line=dict(color=ACCENT, width=3), name="Среднее TVD"))
        fig.add_trace(go.Scatter(x=summary["shots"], y=summary["normal_approx"], mode="lines",
                                 line=dict(color=TEXT, width=1.5, dash="dot"),
                                 name="Ориентир 1/√(2πN)"))
        fig.update_layout(
            title=dict(text="Отклонение частот от вероятностей (TVD)", font=dict(size=16)),
            xaxis=dict(type="log", title="Число измерений N", tickvals=list(CONVERGENCE_SHOTS),
                       gridcolor=BORDER),
            yaxis=dict(title="TVD", gridcolor=BORDER, rangemode="tozero"),
        )
        st.plotly_chart(_base_layout(fig, 480), width="stretch")
        st.markdown(
            f"В среднем по {len(CONVERGENCE_SEEDS)} сериям отклонение уменьшается с ростом числа "
            "измерений. Отдельная случайная серия не обязана улучшаться монотонно: в этом расчёте "
            f"в **{non_monotonic} из {len(CONVERGENCE_SEEDS)}** серий TVD хотя бы раз выросло при "
            "увеличении N. Ориентир 1/√(2πN) — нормальное приближение среднего |f − 0,5| для двух "
            "равновероятных исходов; при N = 10 оно грубое."
        )
        st.download_button("Скачать результаты исследования (CSV)", csv_bytes,
                           file_name="quantumlab_shots_vs_tvd.csv", mime="text/csv")


def render_experiments() -> None:
    st.caption("Все таблицы ниже вычисляются тем же движком, что и в лаборатории.")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown(f"#### {BIT_FLIP.title}")
        _preset_block(BIT_FLIP)
    with c2, st.container(border=True):
        st.markdown(f"#### {SUPERPOSITION.title}")
        _preset_block(SUPERPOSITION)
        for note in SUPERPOSITION.notes:
            st.caption(note)

    with st.container(border=True):
        st.markdown("#### Эксперимент 3 — Интерференция")
        left, right = st.columns(2)
        for column, preset in ((left, INTERFERENCE_HH), (right, INTERFERENCE_HZH)):
            with column:
                st.markdown(f"**`{preset.title}`**")
                _preset_block(preset)
                states = run_circuit_with_history(list(preset.operations), 1)
                st.dataframe(
                    pd.DataFrame({
                        "После шага": ["—"] + [operation_label(o) for o in preset.operations],
                        "Состояние": [format_state(s, 1) for s in states],
                    }),
                    hide_index=True, width="stretch",
                )
        st.markdown(INTERFERENCE_EXPLANATION)

    with st.container(border=True):
        st.markdown(f"#### {BELL.title}")
        state = _preset_block(BELL)
        probs = probabilities(state)
        p_q0 = probs[2] + probs[3]
        p_q1 = probs[1] + probs[3]
        counts = sample_counts(probs, 1000, 7)
        st.markdown(
            f"Отдельные кубиты (рассчитано): P(q0 = 1) = {p_q0:.2f}, P(q1 = 1) = {p_q1:.2f}. "
            f"Симуляция 1000 измерений (seed 7): "
            + ", ".join(f"`{lab}` — {c}" for lab, c in zip(basis_labels(2), counts))
            + "."
        )
        for note in BELL.notes:
            st.markdown(f"- {note}")

    render_convergence()


# ---------- Вкладка «О проекте» ----------

def render_about() -> None:
    with st.container(border=True):
        st.markdown(
            f"""
#### Назначение
QuantumLab — учебная виртуальная лаборатория, в которой можно собрать простую квантовую схему,
рассчитать конечное состояние и сравнить теоретические вероятности с результатами
моделируемых измерений.

#### Для кого
Для школьников и студентов, начинающих изучать квантовые вычисления.

#### Модель: вектор состояния
Состояние n кубитов (n = 1 или 2) хранится как нормированный комплексный вектор NumPy
длины 2ⁿ. Начальное состояние — `|0⟩` или `|00⟩`. {BASIS_CONVENTION}
Каждый гейт — унитарная матрица; для двух кубитов однокубитный гейт расширяется тензорным
произведением (q0 — левый множитель). После каждой операции проверяется нормировка
с допуском {NORM_TOLERANCE:g}; ненормированное состояние вызывает ошибку, а не исправляется молча.

#### Гейты и измерение
`X`, `H`, `Z`, `RY(θ)` (угол в градусах) и `CNOT` с любым управляющим кубитом.
Вероятность исхода равна квадрату модуля амплитуды. Измерения моделируются выборкой из этого
распределения генератором NumPy с заданным seed; каждый shot — новая копия состояния.

#### Технологии
Python, Streamlit, NumPy, Plotly, pytest. Работает локально, без внешних сервисов.

#### Методика исследования
Гипотеза: «Интерактивная модель может помочь учащимся исследовать связь между квантовыми
операциями и вероятностями измерения.»
В приложении проведён воспроизводимый вычислительный эксперимент: для схемы `H` при
N = 10, 100, 1000, 10 000 измерений и seed 1…30 вычисляется расстояние полной вариации
между частотами и вероятностями. Проверка самой гипотезы требует отдельного исследования
с участием учащихся; такое исследование здесь не проводилось.

#### Ограничения
"""
        )
        for item in LIMITATIONS:
            st.markdown(f"- {item}")
        st.caption(f"Версия модели: {MODEL_VERSION}")


# ---------- Страница ----------

def main() -> None:
    _init_state()
    st.title("QuantumLab")
    st.markdown(
        f"<p style='color:{MUTED};font-size:1.15rem;margin-top:-0.6rem'>"
        "Виртуальная лаборатория квантовых вычислений</p>",
        unsafe_allow_html=True,
    )
    st.info("Все расчёты выполняются на классическом компьютере путём моделирования вектора "
            "состояния. Приложение не подключается к реальному квантовому оборудованию.")
    render_message()
    lab, exps, about = st.tabs(["Лаборатория", "Эксперименты", "О проекте"])
    with lab:
        render_lab()
    with exps:
        render_experiments()
    with about:
        render_about()


main()
