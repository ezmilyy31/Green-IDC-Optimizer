"""전역 CSS 스타일 주입 모듈"""

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── 메트릭 카드 (라이트/다크 공용) ─────────────── */
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.15);
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: var(--text-color) !important;
            opacity: 0.55;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
            line-height: 1.2;
        }
        [data-testid="stMetricDelta"] p {
            font-size: 0.76rem !important;
        }

        /* ── 사이드바 ─────────────────────────────────── */
        section[data-testid="stSidebar"] > div:first-child {
            border-top: 3px solid #5D90FF;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            color: #5D90FF !important;
            opacity: 1 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
