"""
전고체 배터리 고체 전해질 온도-이온전도도 시뮬레이션
- VS Code + Streamlit + Plotly 기반 자작 시뮬레이션
- 실행: streamlit run solid_electrolyte_streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# 1. 기본 설정
# -----------------------------
st.set_page_config(
    page_title="고체 전해질 온도-이온전도도 시뮬레이션",
    layout="wide",
)

K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K
FONT_FAMILY = "Malgun Gothic, AppleGothic, NanumGothic, Arial"

# 사용자 제공 표 기준 활성화 에너지값
MATERIALS = {
    "LGPS": 0.22,
    "Li2S-P2S5 glass-ceramic": 0.12,
    "Li3YCl6": 0.40,
    "LLZO": 0.35,
    "Li-LISICON": 0.43,
    "PVDF": 0.34,
}

# -----------------------------
# 2. 계산 함수
# -----------------------------
def simulate(
    selected_materials: list[str],
    temp_min_c: int,
    temp_max_c: int,
    temp_step_c: int,
    sigma0_t: float,
    thickness_um: float,
    area_cm2: float,
    current_ma: float,
) -> pd.DataFrame:
    """온도와 소재별 활성화 에너지에 따른 전도도, 내부저항, 전압 손실 계산."""
    if temp_max_c < temp_min_c:
        raise ValueError("최대 온도는 최소 온도보다 크거나 같아야 합니다.")
    if temp_step_c <= 0:
        raise ValueError("온도 간격은 0보다 커야 합니다.")
    if area_cm2 <= 0:
        raise ValueError("전극 면적은 0보다 커야 합니다.")
    if thickness_um <= 0:
        raise ValueError("전해질 두께는 0보다 커야 합니다.")

    temps_c = np.arange(temp_min_c, temp_max_c + 0.1, temp_step_c, dtype=float)
    temps_k = temps_c + 273.15

    thickness_cm = thickness_um * 1e-4  # 1 um = 1e-4 cm
    current_a = current_ma * 1e-3       # mA -> A

    rows = []
    for material in selected_materials:
        ea = MATERIALS[material]

        # Arrhenius형 이온전도도 모델
        # sigma*T = sigma0_t * exp[-Ea/(kB*T)]
        # sigma = (sigma0_t/T) * exp[-Ea/(kB*T)]
        sigma = (sigma0_t / temps_k) * np.exp(-ea / (K_B_EV * temps_k))  # S/cm

        # 고체 전해질 내부저항과 전압 손실
        resistance = thickness_cm / (sigma * area_cm2)  # ohm
        voltage_loss = current_a * resistance           # V

        # 상대 호핑 지표: exp[-Ea/(kB*T)]
        hopping_factor = np.exp(-ea / (K_B_EV * temps_k))

        for t_c, t_k, s, r, v, h in zip(
            temps_c, temps_k, sigma, resistance, voltage_loss, hopping_factor
        ):
            rows.append(
                {
                    "Material": material,
                    "Ea_eV": ea,
                    "Temperature_C": t_c,
                    "Temperature_K": t_k,
                    "Hopping_factor_relative": h,
                    "Ionic_conductivity_S_per_cm": s,
                    "Internal_resistance_ohm": r,
                    "Voltage_loss_V": v,
                    "Inverse_T_K^-1": 1 / t_k,
                    "ln_sigma_T": np.log(s * t_k),
                }
            )

    return pd.DataFrame(rows)


def make_line_figure(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    y_title: str,
    log_y: bool = False,
) -> go.Figure:
    fig = go.Figure()
    for material in df["Material"].unique():
        sub = df[df["Material"] == material]
        fig.add_trace(
            go.Scatter(
                x=sub["Temperature_C"],
                y=sub[y_col],
                mode="lines+markers",
                name=material,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="온도 (℃)",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title="고체 전해질",
        font=dict(family=FONT_FAMILY),
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def make_arrhenius_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for material in df["Material"].unique():
        sub = df[df["Material"] == material].sort_values("Inverse_T_K^-1")
        x = sub["Inverse_T_K^-1"].to_numpy()
        y = sub["ln_sigma_T"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ea_from_slope = -slope * K_B_EV

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"{material} 데이터",
                hovertemplate="1/T=%{x:.5f}<br>ln(σT)=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_fit,
                mode="lines",
                name=f"{material} 선형 피팅, Ea≈{ea_from_slope:.2f} eV",
            )
        )

    fig.update_layout(
        title="아레니우스 플롯: ln(σT) - 1/T",
        xaxis_title="1/T (K⁻¹)",
        yaxis_title="ln(σT)",
        hovermode="closest",
        legend_title="물질 / 선형 피팅",
        font=dict(family=FONT_FAMILY),
    )
    return fig


def save_outputs(df: pd.DataFrame, figures: dict[str, go.Figure]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    df.to_csv("results/solid_electrolyte_simulation_results.csv", index=False, encoding="utf-8-sig")

    for name, fig in figures.items():
        fig.write_html(f"figures/{name}.html")
        try:
            fig.write_image(f"figures/{name}.png", scale=2)
        except Exception as exc:
            # kaleido 설치/환경 문제 발생 시 HTML만 저장
            print(f"PNG 저장 실패: {name} / 원인: {exc}")


# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("전고체 배터리 고체 전해질 온도-이온전도도 시뮬레이션")

st.markdown(
    """
이 시뮬레이션은 고체 전해질에서 Li⁺ 이온이 격자점 사이를 호핑하며 이동한다고 가정하고,  
Arrhenius형 이온전도도 모델을 이용해 온도와 활성화 에너지에 따른 이온전도도, 내부저항, 전압 손실을 계산합니다.

사용한 기본 관계식은 다음과 같습니다.

- **σT = σ₀ exp[-Ea/(kBT)]**
- **R = L/(σA)**
- **V_loss = IR**
"""
)

with st.sidebar:
    st.header("시뮬레이션 조건")

    selected = st.multiselect(
        "비교할 고체 전해질 선택",
        options=list(MATERIALS.keys()),
        default=list(MATERIALS.keys()),
    )

    temp_min = st.number_input("최소 온도 (°C)", value=20, step=5)
    temp_max = st.number_input("최대 온도 (°C)", value=80, step=5)
    temp_step = st.number_input("온도 간격 (°C)", value=10, min_value=1, step=1)

    st.divider()
    st.caption("통제변인")
    sigma0_t = st.number_input(
        "전지수 인자 σ₀ (S·K/cm)",
        value=1000.0,
        min_value=1e-12,
        format="%.6g",
        help="모든 소재에서 동일하게 유지하여 활성화 에너지 차이의 영향을 비교합니다.",
    )
    thickness_um = st.number_input("고체 전해질 두께 L (μm)", value=100.0, min_value=1.0, step=10.0)
    area_cm2 = st.number_input("전극 면적 A (cm²)", value=1.0, min_value=0.01, step=0.1)
    current_ma = st.number_input("전류 I (mA)", value=10.0, min_value=0.001, step=1.0)

if not selected:
    st.warning("비교할 고체 전해질을 하나 이상 선택해 주세요.")
    st.stop()

try:
    df = simulate(
        selected_materials=selected,
        temp_min_c=int(temp_min),
        temp_max_c=int(temp_max),
        temp_step_c=int(temp_step),
        sigma0_t=float(sigma0_t),
        thickness_um=float(thickness_um),
        area_cm2=float(area_cm2),
        current_ma=float(current_ma),
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

st.subheader("활성화 에너지 입력값")
ea_df = pd.DataFrame(
    [{"Material": m, "Ea_eV": MATERIALS[m]} for m in selected]
)
st.dataframe(ea_df, use_container_width=True)

st.subheader("계산 결과 표")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="계산 결과 CSV 다운로드",
    data=csv,
    file_name="solid_electrolyte_simulation_results.csv",
    mime="text/csv",
)

fig_conductivity = make_line_figure(
    df,
    y_col="Ionic_conductivity_S_per_cm",
    title="온도에 따른 Li⁺ 이온전도도 변화",
    y_title="Li⁺ 이온전도도 (S/cm)",
    log_y=True,
)
fig_resistance = make_line_figure(
    df,
    y_col="Internal_resistance_ohm",
    title="온도에 따른 고체 전해질 내부저항 변화",
    y_title="내부저항 (Ω)",
    log_y=True,
)
fig_voltage = make_line_figure(
    df,
    y_col="Voltage_loss_V",
    title="온도에 따른 전압 손실 변화",
    y_title="전압 손실 (V)",
    log_y=True,
)
fig_hopping = make_line_figure(
    df,
    y_col="Hopping_factor_relative",
    title="온도에 따른 상대 호핑 지표 변화",
    y_title="상대 호핑 지표",
    log_y=True,
)
fig_arrhenius = make_arrhenius_figure(df)

st.subheader("시각화 결과")
st.plotly_chart(fig_conductivity, use_container_width=True)
st.plotly_chart(fig_resistance, use_container_width=True)
st.plotly_chart(fig_voltage, use_container_width=True)
st.plotly_chart(fig_hopping, use_container_width=True)
st.plotly_chart(fig_arrhenius, use_container_width=True)

figures = {
    "conductivity_vs_temperature": fig_conductivity,
    "resistance_vs_temperature": fig_resistance,
    "voltage_loss_vs_temperature": fig_voltage,
    "hopping_factor_vs_temperature": fig_hopping,
    "arrhenius_plot": fig_arrhenius,
}

if st.button("결과 CSV와 그래프 파일 저장"):
    save_outputs(df, figures)
    st.success("results 폴더와 figures 폴더에 결과 파일을 저장했습니다. PNG 저장이 실패하면 HTML 파일을 열어 캡처하면 됩니다.")

st.info(
    "본 시뮬레이션은 활성화 에너지 차이가 Li⁺ 이온 이동에 미치는 영향을 비교하기 위한 단순화 모델입니다. "
    "실제 고체 전해질의 이온전도도는 결정 구조, 도핑, 계면 저항, 결정립계, 제조 조건 등에 따라 달라질 수 있습니다."
)
