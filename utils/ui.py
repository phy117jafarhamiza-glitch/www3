# -*- coding: utf-8 -*-
"""مساعدات واجهة Streamlit المشتركة: إعداد الصفحة، اتجاه RTL، قراءة ملفات CSV."""
import streamlit as st

from db import init_db

RTL_CSS = """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
[data-testid="stSidebar"] { direction: rtl; }
.stTabs [data-baseweb="tab-list"] { direction: rtl; }
</style>
"""


def setup(title: str, icon: str):
    """تهيئة الصفحة. يُستدعى set_page_config مرة واحدة فقط من app.py (st.navigation)."""
    init_db()
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    if not st.session_state.get("_embedded"):
        st.title(f"{icon} {title}")


def embed(path: str):
    """تشغيل صفحة داخل تبويب صفحة أخرى (دون تكرار العنوان)."""
    import runpy
    st.session_state["_embedded"] = True
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        st.session_state["_embedded"] = False


def subject_picker(subjects, label="اللاعب", key=None, allow_none=False):
    """يعيد (subject_dict أو None). يعرض الرمز والاسم والمجموعة."""
    opts = {f'#{s["id"]} {s.get("code") or ""} {s["name"]} — {s["group_name"]}'.replace("  ", " "): s for s in subjects}
    labels = (["— بدون ربط —"] if allow_none else []) + list(opts.keys())
    choice = st.selectbox(label, labels, key=key)
    return opts.get(choice)


def synthetic_warning(is_synth: bool, what: str):
    if is_synth:
        st.warning(f"⚠️ {what} مُدرَّب حاليًا على بيانات تركيبية (للعرض والاختبار فقط). لا تستشهد بنتائجه في الأطروحة "
                   "قبل تدريبه على بيانات حقيقية موسومة من الخبراء (صفحة «تدريب والتحقق من النماذج»).")
