# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import (add_subject, db_bytes, delete_measurement, delete_subject, export_excel_bytes, get_measurements,
                list_subjects, restore_db, upsert_measurement)
from utils import excel_io
from utils.constants import GROUPS, STAGES, TEST_META, TEST_NAMES
from utils.randomize import permuted_block_sequence
from utils.ui import setup, subject_picker

setup("القياسات", "📋")
st.caption("تسجيل اللاعبين وإدخال القياسات T0–T3 يدويًا أو استيرادها من Excel، مع النسخ الاحتياطي.")

t1, t2, t3, t4, t5 = st.tabs(["👤 اللاعبون", "✍️ إدخال القياسات", "📥 استيراد من Excel", "📊 جدول البيانات", "💾 نسخ احتياطي"])

# ------------------------------------------------------------------- اللاعبون
with t1:
    with st.expander("🎲 مولّد التوزيع العشوائي (كتل متبدلة) — يُنفَّذ من شخص غير مشارك في القياس"):
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("عدد المشاركين المخطَّط", 3, 300, 54)
        block = c2.selectbox("حجم الكتلة", [3, 6, 9], index=1)
        seed = c3.number_input("البذرة (Seed) — يحتفظ بها المنفِّذ سرًّا", 0, 10**6, 2026)
        if st.button("توليد التسلسل"):
            seq = permuted_block_sequence(int(n), int(block), int(seed))
            df_seq = pd.DataFrame({"رقم المشارك": range(1, len(seq) + 1), "المجموعة": seq})
            st.dataframe(df_seq, use_container_width=True, hide_index=True)
            st.download_button("⬇️ تنزيل التسلسل (CSV)", df_seq.to_csv(index=False).encode("utf-8-sig"), "allocation.csv")
    st.subheader("إضافة لاعب")
    with st.form("add_subject_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        code = c1.text_input("الرمز (اختياري)", placeholder="P001")
        name = c2.text_input("الاسم")
        group = c3.selectbox("المجموعة", GROUPS)
        sport = c4.text_input("الرياضة القتالية")
        c5, c6, c7, c8, c9 = st.columns(5)
        age = c5.number_input("العمر", 16, 60, 22)
        height = c6.number_input("الطول (سم)", 120.0, 220.0, 175.0)
        weight = c7.number_input("الوزن (كغم)", 40.0, 160.0, 75.0)
        inj_days = c8.number_input("مدة الإصابة (يوم)", 0, 365, 30)
        tr_years = c9.number_input("سنوات التدريب", 0.0, 40.0, 4.0)
        note = st.text_area("ملاحظات الإصابة / التشخيص السريري")
        if st.form_submit_button("إضافة"):
            if not name.strip():
                st.error("أدخل اسم اللاعب.")
            else:
                add_subject(name.strip(), group, int(age), sport.strip(), note.strip(), float(height), float(weight),
                            int(inj_days), float(tr_years), code.strip() or None)
                st.success(f"أُضيف: {name}")
                st.rerun()
    subjects = list_subjects()
    if subjects:
        df = pd.DataFrame(subjects)
        cols = st.columns(len(GROUPS))
        for c, g in zip(cols, GROUPS):
            c.metric(g, int((df["group_name"] == g).sum()))
        st.dataframe(df.drop(columns=["created_at"], errors="ignore"), use_container_width=True, hide_index=True)
        with st.expander("حذف لاعب"):
            opts = {f'#{s["id"]} {s["name"]}': s["id"] for s in subjects}
            pick = st.selectbox("اختر", list(opts.keys()))
            if st.button("حذف نهائي (يحذف كل بياناته)", type="primary"):
                delete_subject(opts[pick])
                st.rerun()
    else:
        st.info("لا يوجد لاعبون بعد.")

# ------------------------------------------------------------ إدخال القياسات
GROUPS_UI = {
    "القوة (كغم)": [k for k in TEST_META if k.startswith("str_")],
    "المدى الحركي (°)": [k for k in TEST_META if k.startswith("rom_")],
    "الألم والخوف الحركي": ["nprs_move", "nprs_rest", "tsk11", "kin_guard"],
    "EMG (% MVC)": ["emg_scm", "emg_scalene", "emg_trap"],
    "الاستجابة الحركية": ["rt_ms", "mt_ms", "rt_acc"],
}
with t2:
    subjects = list_subjects()
    if not subjects:
        st.warning("أضف لاعبين أولًا.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            subj = subject_picker(subjects)
        stage = c2.selectbox("نقطة القياس", STAGES)
        existing = {m["test_key"]: m["value"] for m in get_measurements(subject_id=subj["id"], stage=stage)}
        with st.form("measure_form"):
            values = {}
            tabs = st.tabs(list(GROUPS_UI.keys()))
            for tab, (gname, keys) in zip(tabs, GROUPS_UI.items()):
                with tab:
                    cols = st.columns(3)
                    for i, k in enumerate(keys):
                        name_, unit, lo, hi, _ = TEST_META[k]
                        values[k] = cols[i % 3].number_input(f"{name_} [{unit}]", min_value=lo, max_value=hi,
                                                              value=existing.get(k), step=0.1, key=f"{k}_{subj['id']}_{stage}")
            notes = st.text_input("ملاحظات (اختياري)")
            if st.form_submit_button("💾 حفظ"):
                n = 0
                for k, v in values.items():
                    if v is not None:
                        upsert_measurement(subj["id"], stage, k, float(v), notes or None)
                        n += 1
                st.success(f"حُفظ {n} قياسًا.")
        recs = get_measurements(subject_id=subj["id"])
        if recs:
            df = pd.DataFrame(recs)
            df["المتغير"] = df["test_key"].map(TEST_NAMES)
            piv = df.pivot_table(index="المتغير", columns="stage", values="value", aggfunc="first")
            st.dataframe(piv[[s for s in STAGES if s in piv.columns]], use_container_width=True)
            with st.expander("حذف قياس"):
                keys = sorted({(r["stage"], r["test_key"]) for r in recs})
                pick = st.selectbox("القياس", keys, format_func=lambda x: f"{x[0]} — {TEST_NAMES.get(x[1], x[1])}")
                if st.button("حذف القياس"):
                    delete_measurement(subj["id"], pick[0], pick[1])
                    st.rerun()

# ---------------------------------------------------------------- استيراد Excel
with t3:
    st.markdown("""**استيراد اللاعبين والقياسات من ملف Excel/CSV:** نزّل القالب، املأه (صف لكل لاعب في كل نقطة قياس)، ثم ارفعه.
يُعرَّف اللاعب برمزه أو اسمه، ويُنشأ تلقائيًا إن لم يكن موجودًا (يلزم الاسم والمجموعة). تُراجَع البيانات قبل الكتابة.""")
    st.download_button("⬇️ تنزيل قالب Excel", excel_io.template_bytes(), "neck_rehab_template.xlsx")
    up = st.file_uploader("ملف Excel أو CSV", type=["xlsx", "xls", "csv"], key="main_import")
    create = st.checkbox("إنشاء اللاعبين غير الموجودين تلقائيًا", True)
    if up is not None:
        try:
            raw = excel_io.read_any(up)
            rows, problems = excel_io.parse(raw)
            for p in problems:
                st.warning(p)
            st.write(f"صفوف صالحة للاستيراد: **{len(rows)}** من {len(raw)}")
            if rows:
                prev = pd.DataFrame([{"سطر": r["line"], "اللاعب": r["subject"].get("code") or r["subject"].get("name"),
                                      "نقطة القياس": r["stage"], "عدد المتغيرات": len(r["values"])} for r in rows[:50]])
                st.dataframe(prev, hide_index=True)
                if st.button("✅ استيراد إلى قاعدة البيانات", type="primary"):
                    summ, probs = excel_io.import_rows(rows, create_missing=create)
                    st.success(" — ".join(f"{k}: {v}" for k, v in summ.items()))
                    for p in probs:
                        st.warning(p)
        except Exception as e:
            st.error(f"تعذّرت قراءة الملف: {e}")

# --------------------------------------------------------------------- الجدول
with t4:
    meas = get_measurements()
    subjects = list_subjects()
    if not meas:
        st.info("لا توجد قياسات بعد.")
    else:
        df = pd.DataFrame(meas)
        sdf = pd.DataFrame(subjects)[["id", "code", "name", "group_name"]].rename(columns={"id": "subject_id"})
        wide = df.pivot_table(index=["subject_id", "stage"], columns="test_key", values="value", aggfunc="first").reset_index()
        wide = sdf.merge(wide, on="subject_id")
        st.dataframe(wide, use_container_width=True, hide_index=True)
        st.download_button("⬇️ تنزيل Excel (يتضمن ورقة جاهزة للتحليل)", export_excel_bytes(), "neck_rehab_export.xlsx")

# --------------------------------------------------------------------- نسخ
with t5:
    st.error("⚠️ على Streamlit Community Cloud يكون نظام الملفات مؤقتًا: قد تُفقد البيانات عند إعادة تشغيل التطبيق. "
             "نزّل نسخة احتياطية بعد كل جلسة عمل، وأعد رفعها عند الحاجة.")
    st.download_button("⬇️ نسخة قاعدة البيانات (.db)", db_bytes(), "neck_rehab_backup.db")
    up = st.file_uploader("استعادة من ملف .db", type=["db"], key="restore_db")
    if up is not None and st.button("استعادة (تستبدل البيانات الحالية)", type="primary"):
        restore_db(up.read())
        st.success("تمت الاستعادة.")
