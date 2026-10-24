# -*- coding: utf-8 -*-
import io
import os
import random
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st

from db import add_emg_analysis, add_reaction_trials, get_measurements, list_subjects, upsert_measurement
from ml import webrtc_proc
from ml.reaction import SCREEN_ARROW, STIMULI, analyze_trial, summarize
from ml.video_rom import guarding_index, rom_from_timeline
from utils import excel_io
from utils.constants import DIRECTION_AR, DIRECTIONS, STAGES, TEST_NAMES
from utils.kinematics import symmetry_index
from utils.ui import setup, subject_picker

setup("الاختبارات", "🧪")
st.caption("ستة اختبارات: القوة العضلية، المدى الحركي (كاميرا) مع الحذر الحركي، الألم، EMG، وسرعة الاستجابة الحركية.")

subjects = list_subjects()
if not subjects:
    st.warning("أضف لاعبين أولًا من واجهة «القياسات».")
    st.stop()
c1, c2 = st.columns(2)
with c1:
    subj = subject_picker(subjects)
stage = c2.selectbox("نقطة القياس", STAGES)
cur = {m["test_key"]: m["value"] for m in get_measurements(subject_id=subj["id"], stage=stage)}


def save(vals, label="القياسات"):
    n = 0
    for k, v in vals.items():
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            upsert_measurement(subj["id"], stage, k, float(v))
            n += 1
    st.success(f"حُفظ {n} من {label} للاعب {subj['name']} ({stage}).")


@st.fragment(run_every=0.4)
def _rom_status(proc):
    kind_, msg_ = proc.status()
    a_ = proc.current_angles()
    st.markdown(msg_ + (f"  \nثني {a_['pitch']:.0f}° | ميل {a_['roll']:.0f}° | دوران {a_['yaw']:.0f}°" if a_ else ""))


tabs = st.tabs(["1) القوة العضلية", "2) المدى الحركي + الخوف الحركي", "3) الألم", "4) EMG", "5) سرعة الاستجابة الحركية"])

# ------------------------------------------------------------------ 1) القوة
with tabs[0]:
    st.markdown("القوة الإيزومترية القصوى بالدينامومتر اليدوي (كغم). أدخل ثلاث محاولات لكل اتجاه؛ يُحفظ المتوسط، أو أدخل القيمة النهائية في المحاولة الأولى فقط.")
    means = {}
    cols = st.columns(3)
    for i, d in enumerate(DIRECTIONS):
        key = f"str_{d}"
        with cols[i % 3]:
            st.markdown(f"**{DIRECTION_AR[d]}**" + (f" — محفوظ: {cur[key]:g}" if key in cur else ""))
            tr = [st.number_input(f"محاولة {j + 1}", 0.0, 80.0, None, step=0.1, key=f"s_{d}_{j}_{subj['id']}_{stage}", label_visibility="collapsed" if j else "visible") for j in range(3)]
            tr = [x for x in tr if x is not None]
            if tr:
                means[key] = float(np.mean(tr))
    if means:
        st.dataframe(pd.DataFrame({"الاتجاه": [TEST_NAMES[k] for k in means], "المتوسط (كغم)": [round(v, 2) for v in means.values()]}), hide_index=True)
    allv = {**cur, **means}
    info = []
    if "str_lat_r" in allv and "str_lat_l" in allv:
        info.append(f"عدم تماثل الميل: {symmetry_index(allv['str_lat_r'], allv['str_lat_l']):.1f}%")
    if "str_rot_r" in allv and "str_rot_l" in allv:
        info.append(f"عدم تماثل الدوران: {symmetry_index(allv['str_rot_r'], allv['str_rot_l']):.1f}%")
    if allv.get("str_ext") and "str_flex" in allv:
        info.append(f"نسبة المثنيات/الباسطات: {allv['str_flex'] / allv['str_ext']:.2f}")
    if info:
        st.info(" — ".join(info))
    if means and st.button("💾 حفظ القوة", key="save_str"):
        save(means, "قياسات القوة")

# ------------------------------------------- 2) المدى + الحذر الحركي (فيديو)
with tabs[1]:
    st.markdown("""**طريقة التسجيل:** ابدأ ثابتًا في الوضع المحايد لثانيتين، ثم نفّذ الحركات ببطء مريح بالترتيب.
- **لقطة جانبية:** ثني أمامي ← عودة ← بسط خلفي ← عودة.
- **لقطة أمامية:** ميل يمين ← يسار، ثم دوران يمين ← يسار.

تُحلَّل اللقطتان على حدة (الجانبية للثني والبسط، والأمامية للميل والدوران). الزوايا من كاميرا 2D تقريبية، والدوران أقل دقة.
**الخوف الحركي** يُقدَّر من الفيديو نفسه بمؤشر سلوكي (بطء، توقف وتردد، تقييد المدى، رفع الكتف)؛ هو مؤشر تجريبي
لم تثبت صلاحيته، فاستخدمه مع استبيان TSK-11 (تحت) للمقارنة.""")
    st.session_state.setdefault("rom_results", {})
    mode = st.radio("المصدر", ["رفع فيديو", "بث مباشر"], horizontal=True)
    if mode == "رفع فيديو":
        view_hint = st.selectbox("نوع اللقطة", ["تحديد تلقائي", "جانبية", "أمامية"])
        up = st.file_uploader("فيديو الحركات (mp4/mov)", type=["mp4", "mov", "avi", "m4v"], key="romvid")
        if up is not None and st.button("▶️ تحليل الفيديو", key="run_rom_vid"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
                f.write(up.read())
                path = f.name
            try:
                from ml.live_coach import track_video
                from ml.neck_pose import PoseEstimator
                bar = st.progress(0.0)
                tl = track_video(path, PoseEstimator(), stride=2, progress=bar.progress)
                v = {"جانبية": "side", "أمامية": "front"}.get(view_hint)
                r = rom_from_timeline(tl, view=v)
                g = guarding_index(tl, r["rom"], r["view"])
                st.session_state["rom_results"][r["view"] or "؟"] = (r, g)
            except Exception as e:
                st.error(f"تعذّر التحليل: {e}")
            finally:
                os.unlink(path)
    else:
        if not webrtc_proc.available():
            st.warning("المكتبتان streamlit-webrtc وav غير مثبتتين؛ استخدم «رفع فيديو».")
        else:
            from streamlit_webrtc import webrtc_streamer
            ctx = webrtc_streamer(key="rom_live", video_processor_factory=webrtc_proc.get_processor_class(),
                                  rtc_configuration=webrtc_proc.ICE, media_stream_constraints={"video": True, "audio": False},
                                  async_processing=True)
            if ctx.video_processor:
                ctx.video_processor.configure(None, mirror=True)
                b1, b2 = st.columns(2)
                if b1.button("🔄 إعادة التسجيل من الصفر", key="rom_reset"):
                    ctx.video_processor.reset()
                if b2.button("⏹️ إنهاء وتحليل التسجيل", key="rom_stop"):
                    tl = ctx.video_processor.timeline()
                    r = rom_from_timeline(tl)
                    g = guarding_index(tl, r["rom"], r["view"])
                    st.session_state["rom_results"][r["view"] or "؟"] = (r, g)
                if ctx.video_processor.err:
                    st.error(f"تعذّر تشغيل نموذج الوضعية: {ctx.video_processor.err}")
                st.caption("حرّك رأسك ببطء في كل اتجاه ثم اضغط «إنهاء وتحليل». يجب أن تظهر الأرقام أدناه تتغير مع حركتك؛ إن لم تتغير فالتتبع لا يعمل.")
                _rom_status(ctx.video_processor)
    res = st.session_state["rom_results"]
    if res:
        st.subheader("النتائج")
        merged, guard_vals = {}, []
        for view, (r, g) in res.items():
            st.markdown(f"**لقطة {'جانبية' if view == 'side' else 'أمامية'}** — " + "، ".join(r["notes"]))
            st.dataframe(pd.DataFrame({"الاتجاه": [DIRECTION_AR[k] for k in r["rom"]], "المدى (°)": [round(v, 1) for v in r["rom"].values()]}), hide_index=True)
            for k, v in r["rom"].items():
                merged[f"rom_{k}"] = v
            if g["index"] is not None:
                guard_vals.append(g["index"])
                with st.expander(f"مكوّنات مؤشر الحذر الحركي ({view})"):
                    st.dataframe(pd.DataFrame({"المكوّن": list(g["components"]), "الدرجة": [round(v) for v in g["components"].values()]}), hide_index=True)
        if guard_vals:
            gi = float(np.mean(guard_vals))
            merged["kin_guard"] = gi
            st.metric("مؤشر الحذر الحركي (0–100، الأعلى = حذر أكبر)", f"{gi:.0f}")
        c1_, c2_ = st.columns(2)
        if merged and c1_.button("💾 حفظ المدى ومؤشر الحذر", key="save_rom"):
            save(merged, "قياسات المدى والحذر الحركي")
        if c2_.button("مسح النتائج", key="clear_rom"):
            st.session_state["rom_results"] = {}
            st.rerun()
    st.divider()
    st.markdown("**استبيان الخوف الحركي TSK-11** (مجموع الدرجات 11–44، الأعلى = خوف أكبر)")
    tsk = st.number_input("مجموع TSK-11", 11.0, 44.0, cur.get("tsk11"), step=1.0, key="tsk_in")
    if st.button("💾 حفظ TSK-11", key="save_tsk") and tsk is not None:
        save({"tsk11": tsk}, "الاستبيان")

# ---------------------------------------------------------------- 3) الألم
with tabs[2]:
    c = st.columns(2)
    pm = c[0].select_slider("الألم أثناء الحركة (NPRS 0–10)", options=list(range(0, 11)), value=int(cur.get("nprs_move", 0)), key="pm")
    pr = c[1].select_slider("الألم في الراحة (NPRS 0–10)", options=list(range(0, 11)), value=int(cur.get("nprs_rest", 0)), key="pr")
    if st.button("💾 حفظ الألم", key="save_pain"):
        save({"nprs_move": float(pm), "nprs_rest": float(pr)}, "قياسات الألم")
    st.divider()
    st.markdown("**استيراد الألم من Excel** — أعمدة: رمز اللاعب (أو الاسم)، نقطة القياس، ألم أثناء الحركة، ألم الراحة.")
    tmpl = pd.DataFrame({"رمز اللاعب": ["P001", "P002"], "نقطة القياس": [STAGES[0], STAGES[0]],
                         TEST_NAMES["nprs_move"]: [5, 6], TEST_NAMES["nprs_rest"]: [2, 3]})
    buf = io.BytesIO()
    tmpl.to_excel(buf, index=False)
    st.download_button("⬇️ قالب الألم", buf.getvalue(), "pain_template.xlsx", key="pain_tmpl")
    up = st.file_uploader("ملف الألم (xlsx/csv)", type=["xlsx", "xls", "csv"], key="pain_up")
    if up is not None:
        try:
            rows, problems = excel_io.parse(excel_io.read_any(up))
            for pmsg in problems:
                st.warning(pmsg)
            st.write(f"صفوف صالحة: {len(rows)}")
            if rows and st.button("استيراد الألم", key="pain_imp"):
                summ, pr2 = excel_io.import_rows(rows, create_missing=False)
                st.success(str(summ))
                for x in pr2:
                    st.warning(x)
        except Exception as e:
            st.error(f"تعذّرت قراءة الملف: {e}")

# ------------------------------------------------------------------ 4) EMG
with tabs[3]:
    st.markdown("استيراد إشارات EMG الخام من Excel/CSV: عمود لكل عضلة (وعمود وقت اختياري). تُرشَّح (20–450 Hz + قاطع 50 Hz)، "
                "وتُحسب شدة النشاط كنسبة من MVC، وتُصنَّف النوافذ بشبكة EMG-Net.")
    from ml.emg_cnn import get_classifier
    from ml.signal_processing import bandpass_filter, full_pipeline, notch_filter, rms_envelope
    clf = get_classifier()
    if clf.is_synthetic:
        st.warning("EMG-Net مُدرَّبة حاليًا على بيانات تركيبية (للعرض فقط)؛ نسبة %MVC تبقى صالحة لأنها لا تعتمد على الشبكة.")
    tdf = pd.DataFrame({"time": np.arange(0, 5, 0.001), "SCM": np.zeros(5000), "scalene": np.zeros(5000), "trapezius": np.zeros(5000)})
    b2 = io.BytesIO()
    tdf.to_excel(b2, index=False)
    st.download_button("⬇️ قالب EMG (Excel)", b2.getvalue(), "emg_template.xlsx", key="emg_tmpl")
    fs = st.number_input("تردد العينة fs (Hz)", 250, 4000, 1000, step=50, key="emg_fs")
    up_task = st.file_uploader("إشارة المهمة (أثناء الاختبار)", type=["xlsx", "xls", "csv"], key="emg_task")
    up_mvc = st.file_uploader("إشارة MVC (أقصى انقباض) — اختياري إن أدخلت القيمة يدويًا", type=["xlsx", "xls", "csv"], key="emg_mvc")
    mvc_manual = st.number_input("أو RMS عند MVC لكل عضلة (قيمة واحدة تُطبَّق على الجميع)", 0.0, 1e6, 0.0, key="emg_mvc_val")
    if up_task is not None:
        try:
            dft = excel_io.read_any(up_task)
            num_cols = [c for c in dft.columns if pd.api.types.is_numeric_dtype(dft[c]) and str(c).lower() not in ("time", "t", "الوقت")]
            dfm = excel_io.read_any(up_mvc) if up_mvc is not None else None
            MUS = {"emg_scm": "القصية الترقوية الخشائية", "emg_scalene": "الشقرافية الأمامية", "emg_trap": "شبه المنحرفة العلوية"}
            st.markdown("**ربط الأعمدة بالعضلات**")
            colsel = {}
            cc = st.columns(3)
            for (k, lab), cbox in zip(MUS.items(), cc):
                colsel[k] = cbox.selectbox(lab, ["— لا شيء —"] + num_cols, index=(1 + list(MUS).index(k)) if len(num_cols) > list(MUS).index(k) else 0, key=f"emgcol_{k}")
            if st.button("▶️ تحليل EMG", key="emg_run"):
                out_vals, rows = {}, []
                for k, col in colsel.items():
                    if col == "— لا شيء —":
                        continue
                    sig = dft[col].dropna().to_numpy(float)
                    ref = None
                    if dfm is not None and col in dfm.columns:
                        m = notch_filter(bandpass_filter(dfm[col].dropna().to_numpy(float), fs), fs)
                        ref = float(np.max(rms_envelope(m, fs)))
                    elif mvc_manual > 0:
                        ref = float(mvc_manual)
                    pipe = full_pipeline(sig, fs, ref_amp=ref)
                    pct = 100.0 * float(np.mean(pipe["envelope"])) / pipe["ref_amp"]
                    res_w, major, qn = clf.predict_windows(pipe["raw_windows"], pipe["segments"])
                    out_vals[k] = pct
                    rows.append({"العضلة": MUS[k], "% MVC (متوسط الغلاف)": round(pct, 1),
                                 "تصنيف EMG-Net (الغالب)": major, "احتمال «طبيعي»": round(qn, 2),
                                 "المرجع": "MVC مُدخَل" if ref and (dfm is not None or mvc_manual > 0) else "تقديري (المئين 95)"})
                    add_emg_analysis(subj["id"], stage, MUS[k], getattr(up_task, "name", "emg"), major,
                                     float(np.mean([r["confidence"] for r in res_w])) if res_w else 0.0,
                                     {"rms": float(np.mean(pipe["envelope"])), "mav": None, "zc": None, "wl": None}, clf.backend)
                st.session_state["emg_out"] = (out_vals, rows)
        except Exception as e:
            st.error(f"تعذّرت معالجة الملف: {e}")
    if "emg_out" in st.session_state:
        out_vals, rows = st.session_state["emg_out"]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if any(r["المرجع"].startswith("تقديري") for r in rows):
            st.warning("بعض النتائج بمرجع تقديري؛ أدخل MVC حقيقية لنتائج صالحة للتحليل.")
        if st.button("💾 حفظ نسب %MVC كقياسات", key="emg_save"):
            save(out_vals, "قياسات EMG")

# ----------------------------------------------- 5) سرعة الاستجابة الحركية
with tabs[4]:
    st.markdown("يظهر سهم على الشاشة (من منظورك، الشاشة معكوسة كالمرآة) فتحرّك رأسك نحو الاتجاه بأسرع ما يمكن ثم تعود للوضع المحايد. "
                "الحركات الأمامية (دوران/ميل) من لقطة أمامية، والثني/البسط من لقطة جانبية. "
                "**ملاحظة:** الزمن المقاس يشمل تأخر الكاميرا والعرض، فهو للمقارنة بين نقاط القياس بالجهاز نفسه.")
    if not webrtc_proc.available():
        st.warning("هذا الاختبار يحتاج بثًا مباشرًا: ثبّت streamlit-webrtc وav (موجودتان في requirements.txt).")
    else:
        from streamlit_webrtc import webrtc_streamer
        c = st.columns(3)
        picks = c[0].multiselect("اتجاهات المثير", list(STIMULI), default=["turn_r", "turn_l", "tilt_r", "tilt_l"],
                                 format_func=lambda k: STIMULI[k]["ar"])
        n_tr = c[1].number_input("عدد المحاولات", 4, 40, 12)
        thr = c[2].number_input("عتبة بدء الحركة (°)", 2.0, 15.0, 4.0)
        ctx = webrtc_streamer(key="rt_live", video_processor_factory=webrtc_proc.get_processor_class(),
                              rtc_configuration=webrtc_proc.ICE, media_stream_constraints={"video": True, "audio": False},
                              async_processing=True)
        if ctx.video_processor:
            ctx.video_processor.configure(None, mirror=True)
        if ctx.state.playing and ctx.video_processor and picks and st.button("▶️ ابدأ الاختبار", key="rt_start"):
            proc = ctx.video_processor
            ph = st.empty()
            ph.markdown("## قف في الوضع المحايد… سيبدأ الاختبار بعد 3 ثوانٍ")
            time.sleep(3.0)
            order = [picks[i % len(picks)] for i in range(int(n_tr))]
            random.shuffle(order)
            stims = []
            for i, s_ in enumerate(order):
                ph.markdown("## ● استعد")
                time.sleep(random.uniform(1.5, 3.5))
                t_stim = proc.clock()
                ph.markdown(f"# {SCREEN_ARROW[s_]}  {STIMULI[s_]['ar']}")
                time.sleep(2.0)
                ph.markdown("## ○ عُد للوضع المحايد")
                time.sleep(1.0)
                stims.append((i + 1, s_, t_stim))
            ph.markdown("### انتهى الاختبار — جارٍ التحليل…")
            tl = proc.timeline()
            trials = []
            for no, s_, ts in stims:
                sp = STIMULI[s_]
                if ts is None:
                    continue
                r = analyze_trial(tl, ts, sp["axis"], sp["sign"], thr=float(thr))
                r.update({"trial_no": no, "direction": sp["ar"]})
                trials.append(r)
            st.session_state["rt_trials"] = trials
            ph.empty()
    if st.session_state.get("rt_trials"):
        trials = st.session_state["rt_trials"]
        summ = summarize(trials)
        df = pd.DataFrame(trials)[["trial_no", "direction", "rt_ms", "mt_ms", "peak_velocity", "correct", "note"]]
        df.columns = ["المحاولة", "الاتجاه", "زمن الاستجابة (ms)", "زمن الحركة (ms)", "أقصى سرعة (°/ث)", "صحيحة", "ملاحظة"]
        st.dataframe(df.round(0), hide_index=True)
        c = st.columns(3)
        c[0].metric("متوسط زمن الاستجابة (ms)", f"{summ.get('rt_ms', float('nan')):.0f}" if "rt_ms" in summ else "—")
        c[1].metric("دقة الاتجاه (%)", f"{summ['rt_acc']:.0f}")
        c[2].metric("متوسط زمن الحركة (ms)", f"{summ.get('mt_ms', float('nan')):.0f}" if "mt_ms" in summ else "—")
        if st.button("💾 حفظ نتائج الاستجابة", key="rt_save"):
            add_reaction_trials(subj["id"], stage, trials)
            vals = {"rt_acc": summ["rt_acc"]}
            if "rt_ms" in summ:
                vals.update({"rt_ms": summ["rt_ms"], "mt_ms": summ["mt_ms"]})
            save(vals, "قياسات الاستجابة الحركية")
