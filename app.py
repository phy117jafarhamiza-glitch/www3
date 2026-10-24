# -*- coding: utf-8 -*-
"""AI-NeckRehab: نظام متكامل لتأهيل شدّ عضلات الرقبة لدى لاعبي الفنون القتالية (أداة مساندة لأطروحة)."""
import streamlit as st

st.set_page_config(page_title="AI-NeckRehab", page_icon="🧠", layout="wide")

pages = [
    st.Page("views/measurements.py", title="1) القياسات", icon="📋", url_path="measurements", default=True),
    st.Page("views/tests.py", title="2) الاختبارات", icon="🧪", url_path="tests"),
    st.Page("views/program.py", title="3) البرنامج التأهيلي", icon="📚", url_path="program"),
    st.Page("views/coach.py", title="4) تطبيق التمارين بالكاميرا", icon="🎥", url_path="coach"),
    st.Page("views/stats.py", title="5) الإحصاء", icon="📊", url_path="stats"),
]
with st.sidebar:
    st.markdown("**AI-NeckRehab**  \nتأهيل الرقبة لدى لاعبي الفنون القتالية")
    st.caption("1) القياسات ← 2) الاختبارات ← 3) البرنامج الفردي ← 4) التمرين بالكاميرا ← 5) الإحصاء. "
               "على Streamlit Cloud نزّل نسخة احتياطية بعد كل جلسة (واجهة القياسات).")
st.navigation(pages).run()
