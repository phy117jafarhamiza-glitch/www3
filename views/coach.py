# -*- coding: utf-8 -*-
import os
import tempfile
import time

import pandas as pd
import streamlit as st

from db import add_coach_session, get_measurements, latest_program, list_coach_sessions, list_subjects, upsert_session
from ml import webrtc_proc
from ml.live_coach import VIEW_AR, process_video
from utils.constants import STAGES
from utils.ui import setup, subject_picker

setup("تطبيق التمارين بالكاميرا", "🎥")
st.caption("الكاميرا تتبّع رأسك وكتفيك وتعرض أمامك الوصف الحركي الصحيح، وتعدّ التكرارات، وتؤشر عند الخطأ (إطار أحمر وسهم وتنبيه) "
           "حتى تعود للحركة الصحيحة. المدى المستهدف بالدرجات يُحسب من مداك الحركي المقاس ومستوى الأسبوع. "
           "الزوايا من كاميرا 2D تقريبية (خطأ قد يبلغ 5–10°)، والدوران أقل دقة.")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا.")
    st.stop()
subj = subject_picker(subjects)
prog = latest_program(subj["id"])
if not prog:
    st.warning("لا يوجد برنامج تأهيلي لهذا اللاعب. أنشئه من واجهة «البرنامج التأهيلي».")
    st.stop()
P = prog["program"]

c1, c2 = st.columns(2)
week = c1.selectbox("الأسبوع", [w["week"] for w in P["weeks"]])
wk = next(w for w in P["weeks"] if w["week"] == week)
unit_no = c2.selectbox("الوحدة", [u["unit"] for u in wk["units"]],
                       format_func=lambda i: f"الوحدة {i} — {next(u for u in wk['units'] if u['unit'] == i)['name']} "
                                             f"({next(u for u in wk['units'] if u['unit'] == i)['minutes']} د)")
unit = next(u for u in wk["units"] if u["unit"] == unit_no)
entries = [dict(e, _block=b) for b, lab in (("warmup", "إحماء"), ("main", "رئيسي"), ("cooldown", "ختام")) for e in unit.get(b, [])]
cam_entries = [e for e in entries if e.get("cam")]

st.markdown(f"**المستوى {wk['level']} — الوحدة {unit['minutes']} دقيقة.** تمارين الوحدة: " + "، ".join(e["name"] for e in entries))
if not cam_entries:
    st.info("لا توجد في هذه الوحدة تمارين قابلة للتتبع بالكاميرا (التمارين الإيزومترية تُتابَع بالقوة/EMG).")
    st.stop()

ex = st.selectbox("التمرين (قابل للتتبع بالكاميرا)", range(len(cam_entries)),
                  format_func=lambda i: f"{cam_entries[i]['name']} — {cam_entries[i].get('sets', '')}×{cam_entries[i].get('reps', '')}"
                                        + (f" ({cam_entries[i]['side']})" if cam_entries[i].get("side") else ""))
e = cam_entries[ex]
spec = dict(e["cam"])
with st.expander("⚙️ المدى المستهدف بالدرجات (يعدّله الأخصائي)"):
    cc = st.columns(3)
    spec["target"] = cc[0].number_input("المدى المستهدف (°)", 3.0, 120.0, float(spec["target"]))
    spec["safe"] = cc[1].number_input("الحد الآمن (°)", 5.0, 150.0, float(max(spec["safe"], spec["target"] + 2)))
    if spec.get("mode") == "rep":
        spec["min_rep_s"] = cc[2].number_input("أقل زمن للتكرار (ث)", 1.0, 10.0, float(spec.get("min_rep_s", 3.0)))
    else:
        spec["hold_s"] = cc[2].number_input("زمن الثبات (ث)", 3.0, 120.0, float(spec.get("hold_s", 20.0)))
st.info(f"المدى المستهدف: **{spec['target']:.0f}°** — الحد الآمن: **{spec['safe']:.0f}°** — زاوية الكاميرا المطلوبة: "
        f"**{VIEW_AR.get(spec.get('view'), 'أي زاوية')}**. قف أو استلقِ بوضع محايد ثابت لثانيتين للمعايرة ثم ابدأ.")

left, right = st.columns([2, 1])
with right:
    st.subheader("الوصف الحركي الصحيح")
    for i, s_ in enumerate(e.get("instructions", []), 1):
        st.markdown(f"{i}. {s_}")
    if e.get("common_errors"):
        st.markdown("**أخطاء شائعة:**")
        for s_ in e["common_errors"]:
            st.markdown(f"- ⚠️ {s_}")
    if e.get("rationale"):
        st.caption(f"سبب اختيار التمرين لك: {e['rationale']}")
    fb = st.empty()


@st.fragment(run_every=0.4)
def _live(proc):
    """يُحدَّث وحده كل 0.4 ث دون حجب بقية الصفحة."""
    kind, msg = proc.status()
    a_ = proc.current_angles()
    box = st.container()
    box.markdown(msg + (f"  \nثني {a_['pitch']:.0f}° | ميل {a_['roll']:.0f}° | دوران {a_['yaw']:.0f}° | "
                        f"اللقطة: {VIEW_AR.get(a_['view'], '')}" if a_ else ""))
    snap = proc.snapshot()
    if snap and kind == "ok":
        head = f"### التكرارات: {snap['reps']} — {snap['rel']:.0f}° / {snap['target']:.0f}°"
        if snap.get("mode") != "rep" and snap.get("hold_goal"):
            head = f"### الثبات: {snap['hold_time']:.0f} / {snap['hold_goal']:.0f} ث — {snap['rel']:.0f}°"
        errs = set(snap.get("errors", []))
        lines = [(":red[**⛔ " + m + "**]") if m in errs else (":green[**✔ " + m + "**]") for m in snap["messages"]]
        box.markdown(head + "\n\n" + "\n\n".join(lines))


def show(snap):
    if not snap:
        return
    head = f"### التكرارات: {snap['reps']} — {snap['rel']:.0f}° / {snap['target']:.0f}°"
    if snap.get("mode") != "rep" and snap.get("hold_goal"):
        head = f"### الثبات: {snap['hold_time']:.0f} / {snap['hold_goal']:.0f} ث — {snap['rel']:.0f}°"
    errs = set(snap.get("errors", []))
    lines = [(":red[**⛔ " + m + "**]") if m in errs else (":green[**✔ " + m + "**]") for m in snap["messages"]]
    fb.markdown(head + "\n\n" + "\n\n".join(lines))


def save_summary(s):
    add_coach_session(subject_id=subj["id"], week=int(week), session_no=int(unit_no), exercise_id=int(e.get("id", 0)),
                      reps=s["reps"], mean_rom=s["mean_rom"], max_velocity=s["max_velocity"], too_fast=s["too_fast"],
                      short_rom=s["short_rom"], compensation=s["compensation"], over_limit=s["over_limit"],
                      quality_score=s["quality_score"])
    st.success("حُفظ ملخص التمرين.")


tab1, tab2, tab3 = st.tabs(["بث مباشر", "فيديو مسجَّل", "سجل التطبيق"])
with tab1:
    with left:
        if not webrtc_proc.available():
            st.warning("المكتبتان streamlit-webrtc وav غير مثبتتين (موجودتان في requirements.txt). استخدم «فيديو مسجَّل» في الأثناء.")
        else:
            from streamlit_webrtc import webrtc_streamer
            ctx = webrtc_streamer(key="coach", video_processor_factory=webrtc_proc.get_processor_class(),
                                  rtc_configuration=webrtc_proc.ICE, media_stream_constraints={"video": True, "audio": False},
                                  async_processing=True)
            if ctx.video_processor:
                ctx.video_processor.configure(spec, mirror=True)
                b1, b2 = st.columns(2)
                if b1.button("🔄 إعادة المعايرة / تمرين جديد"):
                    ctx.video_processor.reset()
                if b2.button("💾 إنهاء وحفظ ملخص التمرين"):
                    s = ctx.video_processor.summary()
                    if s:
                        save_summary(s)
                        st.json(s)
                if ctx.video_processor.err:
                    st.error(f"تعذّر تشغيل نموذج الوضعية: {ctx.video_processor.err}")
                with right:
                    _live(ctx.video_processor)

with tab2:
    up = st.file_uploader("فيديو للتمرين (mp4/mov)", type=["mp4", "mov", "avi", "m4v"])
    if up is not None and st.button("▶️ تحليل الفيديو"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
            f.write(up.read())
            path = f.name
        try:
            from ml.neck_pose import PoseEstimator
            bar = st.progress(0.0)
            s, last, tl = process_video(path, spec, PoseEstimator(), stride=2, progress=bar.progress)
            st.session_state["vid_sum"] = (s, tl)
            if last is not None:
                st.image(last, caption="آخر إطار محلَّل")
        except Exception as ex_:
            st.error(f"تعذّر التحليل: {ex_}")
        finally:
            os.unlink(path)
    if "vid_sum" in st.session_state:
        s, tl = st.session_state["vid_sum"]
        st.json(s)
        if tl:
            st.line_chart(pd.DataFrame(tl).set_index("t")["angle"])
        if st.button("💾 حفظ ملخص الفيديو"):
            save_summary(s)

with tab3:
    rows = list_coach_sessions(subj["id"])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df.sort_values("id").set_index("id")["quality_score"])
    else:
        st.caption("لا توجد جلسات محفوظة.")
    st.divider()
    st.markdown("**تسجيل حضور الوحدة**")
    cc = st.columns(4)
    mins = cc[0].number_input("الدقائق الفعلية", 0.0, 120.0, float(unit["minutes"]))
    rpe = cc[1].number_input("الجهد المدرك RPE (0–10)", 0.0, 10.0, None)
    done = cc[2].checkbox("اكتملت الوحدة", True)
    if cc[3].button("حفظ الحضور"):
        upsert_session(subj["id"], int(week), int(unit_no), mins, done, rpe)
        st.success("حُفظ.")
