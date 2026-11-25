"""Executive visual theme and reusable Streamlit cards."""
from __future__ import annotations

from html import escape
from typing import Sequence

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        #MainMenu, footer, [data-testid="stAppDeployButton"],
        [data-testid="stToolbar"] {visibility: hidden; height: 0;}
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stSidebar"] {background: #0F172A;}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] summary {color: #E2E8F0;}
        [data-baseweb="select"] > div {
            background: #FFFFFF !important; color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] span {color: #0F172A !important;}
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {
            background: #FFFFFF !important; color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [role="listbox"] {background: #FFFFFF !important;}
        [role="option"] {background: #FFFFFF !important; color: #0F172A !important;}
        [role="option"]:hover {background: #E0F2FE !important;}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #1E293B !important; color: #F8FAFC !important;
            border-color: #334155 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #F8FAFC !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {
            background: #1E293B; color: #F8FAFC; border-color: #334155;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
            background: #F8FAFC !important; color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #334155; border-radius: 10px;
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background: #10B981; border: 1px solid #10B981; color: #052E2B;
            font-weight: 700; border-radius: 9px;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background: #34D399; border-color: #34D399; color: #052E2B;
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: .45rem; border-bottom: 1px solid #E2E8F0;
        }
        [data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px; padding: .45rem 1rem; border: 1px solid transparent;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #FFFFFF; background: #0F766E; border-color: #0F766E;
        }
        .hero-banner {
            padding: 1.35rem 1.6rem; margin: .15rem 0 1.1rem;
            border-radius: 16px; color: #FFFFFF;
            background: linear-gradient(115deg, #0F172A 0%, #134E4A 100%);
            box-shadow: 0 12px 28px rgba(15, 23, 42, .16);
        }
        .hero-title {font-size: 1.7rem; font-weight: 750; line-height: 1.2;}
        .hero-subtitle {margin-top: .35rem; color: #CDE7E2; font-size: .94rem;}
        .hero-pills {display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem;}
        .hero-pill {
            display: inline-flex; gap: .35rem; align-items: center;
            padding: .35rem .7rem; border: 1px solid rgba(226, 232, 240, .24);
            border-radius: 999px; color: #F8FAFC; background: rgba(15, 23, 42, .3);
            font-size: .8rem;
        }
        .kpi-card {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid;
            border-radius: 12px; padding: .9rem 1rem; min-height: 112px;
            box-shadow: 0 5px 16px rgba(15, 23, 42, .05);
        }
        .kpi-label {color: #64748B; font-size: .8rem; font-weight: 650;}
        .kpi-value {color: #0F172A; font-size: 1.4rem; font-weight: 750; margin-top: .3rem;}
        .kpi-context {color: #64748B; font-size: .75rem; margin-top: .3rem;}
        .section-heading {color: #0F172A; font-weight: 700; margin: .4rem 0 .7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    instance: str,
    client_count: int,
    elapsed_seconds: float | None,
    front_size: int,
) -> None:
    elapsed = "Pendiente" if elapsed_seconds is None else f"{elapsed_seconds:.2f} s"
    pills = (
        ("Instancia", instance),
        ("Clientes", str(client_count)),
        ("Tiempo de cómputo", elapsed),
        ("Soluciones Frente 1", str(front_size)),
    )
    badge_html = "".join(
        f'<span class="hero-pill"><b>{escape(label)}:</b> {escape(value)}</span>'
        for label, value in pills
    )
    st.markdown(
        '<section class="hero-banner">'
        '<div class="hero-title">Green-VRPTW Control Tower</div>'
        '<div class="hero-subtitle">Centro de control de ruteo verde · '
        'NSGA-II triobjetivo con ventanas de tiempo</div>'
        f'<div class="hero-pills">{badge_html}</div></section>',
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: Sequence[tuple[str, str, str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, (label, value, context, color) in zip(columns, cards):
        column.markdown(
            '<div class="kpi-card" style="border-left-color:'
            f'{escape(color)}">'
            f'<div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div>'
            f'<div class="kpi-context">{escape(context)}</div></div>',
            unsafe_allow_html=True,
        )
