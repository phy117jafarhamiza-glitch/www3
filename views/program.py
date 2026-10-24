# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from db import get_measurements, latest_program, list_programs, list_sessions, list_subjects, save_program, upsert_session
from utils.constants import STAGES, TEST_NAMES
from utils.program_gen import (DEFAULT_CFG, build_profile, generate_program, program_to_dataframe,
                               program_to_docx_bytes, program_to_xlsx_bytes)
from utils.ui import setup, subject_picker

setup("البرنامج التأهيلي الفردي", "📚")
st.caption("يُبنى لكل لاعب من نتائج اختباراته: 8 أسابيع × 3 وحدات، زمن الوحدة من 15 إلى 90 دقيقة، بتدرج من السهل إلى الصعب. "
           "القواعد والعتبات مبدئية ويعتمدها المشرف/الأخصائي؛ وهو نظام خبير قائم على قواعد وليس تعلّمًا آليًا.")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين من واجهة «القياسات» ثم أجرِ اختباراتهم.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    subj = subject_picker(subjects)
allm = get_measurements(subject_id=subj["id"])
stages_have = [s for s in STAGES if any(m["stage"] == s for m in allm)]
if not stages_have:
    st.warning("لا توجد قياسات لهذا اللاعب. أجرِ الاختبارات أو استوردها أولًا.")
    st.stop()
stage = c2.selectbox("بناء البرنامج على قياسات", stages_have, index=len(stages_have) - 1)
m = {r["test_key"]: r["value"] for r in allm if r["stage"] == stage}

with st.expander(f"نتائج الاختبارات المعتمدة ({len(m)} متغيرًا)", expanded=False):
    st.dataframe(pd.DataFrame({"المتغير": [TEST_NAMES.get(k, k) for k in m], "القيمة": list(m.values())}), hide_index=True)

with st.expander("⚙️ عتبات القواعد (مبدئية — يعدّلها الأخصائي)"):
    cfg = {}
    cols = st.columns(3)
    i = 0
    for k, v in DEFAULT_CFG.items():
        if isinstance(v, (int, float)):
            cfg[k] = cols[i % 3].number_input(k, value=float(v), key=f"cfg_{k}")
            i += 1

if st.button("🧠 توليد البرنامج الفردي", type="primary"):
    profile = build_profile(subj, m, cfg)
    program = generate_program(profile, cfg)
    st.session_state["prog_new"] = (subj["id"], stage, program)

new = st.session_state.get("prog_new")
if new and new[0] == subj["id"]:
    _, stg, program = new
    st.success("تم توليد البرنامج. راجعه ثم احفظه ليظهر في واجهة التمرين بالكاميرا.")
else:
    saved = latest_program(subj["id"])
    program = saved["program"] if saved else None
    stg = saved["based_on_stage"] if saved else None
    if saved:
        st.info(f"آخر برنامج محفوظ (بُني على {stg} — {saved['created_at']}).")

if program:
    prof = program.get("profile", {})
    st.subheader("ملف اللاعب (ما الذي دفع إلى هذا البرنامج؟)")
    findings = prof.get("findings", [])
    if findings:
        st.dataframe(pd.DataFrame([{"المحور": f.get("title"), "الشدة": f.get("severity"), "التفسير": f.get("text")} for f in findings]),
                     use_container_width=True, hide_index=True)
    for pre in prof.get("precautions", []):
        st.warning(pre)
    for n_ in program.get("notes", []):
        st.caption(n_)

    st.subheader("نظرة عامة")
    ov = pd.DataFrame([{"الأسبوع": w["week"], "المستوى": w["level"], "الوحدة 1 (د)": w["unit_minutes"][0],
                        "الوحدة 2 (د)": w["unit_minutes"][1], "الوحدة 3 (د)": w["unit_minutes"][2]} for w in program["weeks"]])
    st.dataframe(ov, use_container_width=True, hide_index=True)
    st.bar_chart(ov.set_index("الأسبوع")[["الوحدة 1 (د)", "الوحدة 2 (د)", "الوحدة 3 (د)"]])

    for w in program["weeks"]:
        with st.expander(f"الأسبوع {w['week']} — المستوى {w['level']}"):
            for u in w["units"]:
                st.markdown(f"**الوحدة {u['unit']} — {u['name']} ({u['minutes']} دقيقة)**")
                rows = []
                for blk, lab in (("warmup", "إحماء"), ("main", "رئيسي"), ("cooldown", "ختام")):
                    for e in u.get(blk, []):
                        rows.append({"الجزء": lab, "التمرين": e["name"] + (f" ({e['side']})" if e.get("side") else ""),
                                     "المجاميع": e.get("sets"), "التكرارات": e.get("reps"), "ثبات (ث)": e.get("hold_s"),
                                     "راحة (ث)": e.get("rest_s"), "مقاومة %MVC": e.get("resistance_pct_mvc"),
                                     "دقائق": e.get("minutes"), "كاميرا": "✓" if e.get("cam") else "",
                                     "سبب الاختيار": e.get("rationale", "")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    b = st.columns(4)
    if new and new[0] == subj["id"]:
        if b[0].button("💾 حفظ البرنامج للاعب"):
            save_program(subj["id"], program, stg)
            st.session_state.pop("prog_new", None)
            st.success("حُفظ.")
            st.rerun()
    b[1].download_button("⬇️ Excel", program_to_xlsx_bytes(program), f"program_{subj['code'] or subj['id']}.xlsx")
    b[2].download_button("⬇️ Word", program_to_docx_bytes(program), f"program_{subj['code'] or subj['id']}.docx")
    b[3].download_button("⬇️ CSV", program_to_dataframe(program).to_csv(index=False).encode("utf-8-sig"),
                         f"program_{subj['code'] or subj['id']}.csv")

    st.divider()
    st.subheader("متابعة تنفيذ الوحدات")
    cc = st.columns(5)
    w_ = cc[0].number_input("الأسبوع", 1, 8, 1)
    u_ = cc[1].number_input("الوحدة", 1, 3, 1)
    mins_default = float(program["weeks"][int(w_) - 1]["unit_minutes"][int(u_) - 1])
    mins = cc[2].number_input("الدقائق الفعلية", 0.0, 120.0, mins_default)
    rpe = cc[3].number_input("RPE (0–10)", 0.0, 10.0, None)
    done = cc[4].checkbox("اكتملت", True)
    if st.button("حفظ الوحدة"):
        upsert_session(subj["id"], int(w_), int(u_), mins, done, rpe)
        st.rerun()
    rows = list_sessions(subj["id"])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["subject_id"]), use_container_width=True, hide_index=True)
        st.metric("الالتزام (الوحدات المكتملة من 24)", f"{100 * df['completed'].sum() / 24:.0f}%")
    st.info("لإعادة تفصيل البرنامج بعد إعادة الاختبار (T1/T2) اختر نقطة القياس ثم ولّده من جديد؛ يبقى كل برنامج سابق محفوظًا.")
else:
    st.info("اضغط «توليد البرنامج الفردي».")
