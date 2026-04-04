"""
AI Green IDC 대시보드 — 진입점

st.navigation()으로 페이지를 명시적으로 등록한다.
st.set_page_config()는 여기서 한 번만 호출하고 개별 페이지에서는 호출하지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(
    page_title="AI Green IDC Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages/0_대시보드.py",  title="대시보드",  icon=":material/dashboard:",  default=True),
        st.Page("pages/1_운영_관리.py", title="운영 관리", icon=":material/tune:"),
        st.Page("pages/2_ESG_지표.py",  title="ESG 지표",  icon=":material/eco:"),
        st.Page("pages/3_분석_도구.py", title="분석 도구", icon=":material/analytics:"),
    ],
    position="sidebar",
)

pg.run()
