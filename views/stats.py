# -*- coding: utf-8 -*-
"""صفحة الإحصاء — بديل SPSS: معلمي / لا معلمي / تلقائي، من قاعدة التطبيق أو من ملف Excel/CSV."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from db import get_measurements, list_subjects
from utils import stats as S
from utils import stats2 as S2
from utils.constants import COVARIATES, PRIMARY_KEY, STAGES, TEST_META, TEST_NAMES
from utils.ui import setup

setup("الإحصاء", "📊")
st.caption("بديل كامل عن SPSS: اختبارات معلمية ولا معلمية مع حجم الأثر وفترات الثقة وتصحيح المقارنات المتعددة، "
           "وتفسير عربي تلقائي لكل نتيجة (مستوى الدلالة α = 0.05). التفسير آلي ومساعد ولا يغني عن حكم الباحث.")

# ------------------------------------------------------------------ نتائج الجلسة
RES_TABLES = {}   # اسم الورقة -> DataFrame
RES_NOTES = []    # (التحليل، التفسير)


def reg(name, df, note=None):
    if isinstance(df, pd.DataFrame) and len(df):
        RES_TABLES[name] = df
    if note:
        RES_NOTES.append((name, note))


def nice(df):
    """تقريب العرض (الأرقام فقط)."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(4)
    return out


def table(df, title=None):
    if isinstance(df, pd.DataFrame) and len(df):
        if title:
            st.markdown(f"**{title}**")
        st.dataframe(nice(df), use_container_width=True)


def verdict(res, p=None):
    """st.success إذا كانت النتيجة دالة وإلا st.info."""
    p = res.get("p") if p is None else p
    if isinstance(p, dict):
        p = min([v for v in p.values() if v is not None and np.isfinite(v)] or [1.0])
    try:
        sig = p is not None and np.isfinite(p) and p < 0.05
    except TypeError:
        sig = False
    (st.success if sig else st.info)(res["interp"])


def guard(res):
    if not res.get("ok"):
        st.warning("⚠️ " + res.get("reason", "تعذّر إجراء التحليل."))
        return False
    return True


# ------------------------------------------------------------------ الإعدادات العامة
MODES = {"إحصاء معلمي (Parametric)": "param", "إحصاء لا معلمي (Non-parametric)": "nonparam",
         "تلقائي (حسب الاعتدالية)": "auto"}
mode_label = st.radio("نوع الإحصاء", list(MODES), horizontal=True,
                      help="التلقائي: يختار المعلمي إذا تحققت الاعتدالية (Shapiro–Wilk) في كل مجموعة، وإلا اللامعلمي.")
FORCE = MODES[mode_label]
ADJ = {"Holm": "holm", "Bonferroni": "bonferroni", "Benjamini–Hochberg (FDR)": "bh"}
c1, c2 = st.columns([1, 1])
adj_label = c1.selectbox("تصحيح المقارنات المتعددة", list(ADJ))
ADJUST = ADJ[adj_label]
c2.info("مستوى الدلالة α = 0.05 (ثنائي الطرف)")

src = st.radio("مصدر البيانات", ["من قاعدة بيانات التطبيق", "رفع ملف Excel/CSV"], horizontal=True)


def to_num(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


df = None
group_col, id_col, time_cols, extras, data_label = None, None, [], [], ""
if src.startswith("من قاعدة"):
    subjects, meas = list_subjects(), get_measurements()
    if len(subjects) < 3 or not meas:
        st.info("لا توجد بيانات كافية في قاعدة التطبيق (أضف لاعبين وقياسات في صفحة القياسات) — أو استخدم خيار رفع ملف Excel/CSV.")
    else:
        keys = list(TEST_META.keys())
        key = st.selectbox("المتغير", keys, format_func=lambda k: TEST_NAMES[k],
                           index=keys.index(PRIMARY_KEY) if PRIMARY_KEY in keys else 0)
        stages = st.multiselect("نقاط القياس المشمولة (بالترتيب الزمني)", STAGES, default=STAGES)
        stages = [s for s in STAGES if s in stages]
        if len(stages) < 1:
            st.warning("اختر نقطة قياس واحدة على الأقل.")
        else:
            df = S.wide_table(meas, subjects, key, stages)
            df = df.dropna(subset=["group_name"])
            if int(df[stages].notna().sum().sum()) < 3:
                st.warning("لا توجد قياسات كافية مسجلة لهذا المتغير في نقاط القياس المختارة.")
                df = None
        if df is not None:
            group_col, id_col, time_cols = "group_name", "subject_id", stages
            sdf = pd.DataFrame(subjects)
            extras = [c for c in COVARIATES if c in sdf.columns]
            if extras:
                df = df.merge(sdf[["id"] + extras].rename(columns={"id": "subject_id"}), on="subject_id", how="left")
            data_label = TEST_NAMES[key]
else:
    up = st.file_uploader("ارفع ملف Excel (.xlsx) أو CSV", type=["xlsx", "xls", "csv"])
    if up is None:
        st.info("ارفع ملفًا ثم حدّد أعمدة المجموعة والمعرّف والقيم.")
    else:
        try:
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up, encoding="utf-8-sig")
            else:
                xl = pd.ExcelFile(up)
                sheet = st.selectbox("الورقة", xl.sheet_names) if len(xl.sheet_names) > 1 else xl.sheet_names[0]
                raw = xl.parse(sheet)
        except Exception as e:  # noqa: BLE001
            raw = None
            st.error(f"تعذّر قراءة الملف: {e}")
        if raw is not None:
            raw = raw.dropna(how="all")
            raw.columns = [str(c).strip() for c in raw.columns]
            st.caption(f"تم تحميل {len(raw)} صفًا و{raw.shape[1]} عمودًا.")
            table(raw.head(10), "معاينة أول 10 صفوف")
            cols_all = list(raw.columns)
            if len(cols_all) < 2:
                st.warning("الملف يحتاج عمودين على الأقل (مجموعة + قيمة).")
            else:
                layout = st.radio("تخطيط البيانات", ["عريض: عمود لكل نقطة قياس (أو عمود قيمة واحد)",
                                                     "طويل: عمود للزمن وعمود للقيمة"], horizontal=True)
                gc = st.selectbox("عمود المجموعة", cols_all)
                idc = st.selectbox("عمود معرّف اللاعب (اختياري في التخطيط العريض)", ["— بدون —"] + cols_all)
                idc = None if idc == "— بدون —" else idc
                if layout.startswith("عريض"):
                    guess = [c for c in cols_all if c not in (gc, idc) and pd.to_numeric(raw[c], errors="coerce").notna().mean() > .5]
                    vals = st.multiselect("أعمدة القيم (بترتيب نقاط القياس؛ عمود واحد = قياس واحد)", [c for c in cols_all if c not in (gc, idc)],
                                          default=guess[:4])
                    if vals:
                        df = to_num(raw, vals)
                        df = df.dropna(subset=[gc])
                        group_col, id_col, time_cols = gc, idc, vals
                        extras = [c for c in guess if c not in vals][:8]
                        if extras:
                            df = to_num(df, extras)
                    else:
                        st.warning("اختر عمود قيمة واحدًا على الأقل.")
                else:
                    tc = st.selectbox("عمود الزمن/نقطة القياس", [c for c in cols_all if c not in (gc, idc)])
                    vc = st.selectbox("عمود القيمة", [c for c in cols_all if c not in (gc, idc, tc)] or cols_all)
                    if idc is None:
                        st.warning("التخطيط الطويل يتطلب عمود معرّف اللاعب.")
                    else:
                        tmp = raw[[gc, idc, tc, vc]].copy()
                        tmp[vc] = pd.to_numeric(tmp[vc], errors="coerce")
                        order = list(pd.unique(tmp[tc].astype(str)))
                        tmp[tc] = tmp[tc].astype(str)
                        piv = tmp.pivot_table(index=[idc, gc], columns=tc, values=vc, aggfunc="first").reset_index()
                        time_cols = [c for c in order if c in piv.columns]
                        df, group_col, id_col = piv.dropna(subset=[gc]), gc, idc
                        st.caption("ترتيب الأزمنة حسب ظهورها في الملف: " + " ← ".join(time_cols))
                if df is not None:
                    nonnum = int(df[time_cols].isna().sum().sum() - raw.reindex(df.index)[time_cols].isna().sum().sum()) if layout.startswith("عريض") and time_cols else 0
                    if nonnum > 0:
                        st.warning(f"تم تجاهل {nonnum} خلية غير رقمية في أعمدة القيم.")
                    data_label = up.name

# ------------------------------------------------------------------ تهيئة البيانات
ready = df is not None and group_col is not None and len(time_cols) >= 1 and len(df) >= 3
delta_cols = []
if ready:
    df = df.copy()
    df[group_col] = df[group_col].astype(str)
    base0 = time_cols[0]
    for c in time_cols[1:]:
        nm = f"Δ ({c}) − ({base0})"
        df[nm] = df[c] - df[base0]
        delta_cols.append(nm)
    groups_all = list(pd.unique(df[group_col]))
    value_cols = time_cols + delta_cols
    numeric_cols = value_cols + [c for c in extras if c in df.columns]
    st.success(f"البيانات: «{data_label}» — {len(df)} لاعبًا، {len(groups_all)} مجموعات "
               f"({'، '.join(groups_all)})، نقاط القياس: {len(time_cols)}.")
    with st.expander("عرض الجدول العريض المحلَّل"):
        table(df)
elif df is not None:
    st.warning("البيانات غير كافية (يلزم 3 صفوف على الأقل وعمود قيمة واحد).")

if not ready:
    st.stop()


def fam_for(groups, label="", resid=None):
    """يحدد الأسرة (معلمي/لا معلمي) ويعرض السبب."""
    if resid is not None:
        fam, reason = S2.choose_family_residuals(resid, FORCE)
    else:
        fam, reason = S2.choose_family(groups, FORCE)
    name = "معلمي" if fam == "param" else "لا معلمي"
    st.caption(f"الأسلوب المعتمد{(' لـ ' + label) if label else ''}: **{name}** — {reason}")
    return fam


def groups_of(col, names=None):
    out = {}
    for g in (names or groups_all):
        v = pd.to_numeric(df.loc[df[group_col] == g, col], errors="coerce").dropna().to_numpy(float)
        out[g] = v
    return out


def wide_clean(cols):
    return df[[group_col] + list(cols)].dropna()


def means_chart(cols, title="متوسط كل مجموعة عبر نقاط القياس"):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    m = df.groupby(group_col, sort=False)[cols].mean().T
    m.index = [f"{i + 1}) {c}" for i, c in enumerate(cols)]
    st.markdown(f"**{title}**")
    st.line_chart(m)


tabs = st.tabs(["الوصفي والافتراضات", "مقارنة مجموعتين", "مقارنة عدة مجموعات", "القياسات المتكررة (داخل المجموعة)",
                "التصميم المختلط (مجموعة × زمن) + ANCOVA", "الارتباط والانحدار", "الجداول التكرارية", "حاسبة حجم العينة والقوة"])

# ================================================================== 1) الوصفي والافتراضات
with tabs[0]:
    st.subheader("الإحصاء الوصفي")
    sel = st.multiselect("المتغيرات/القياسات", numeric_cols, default=time_cols, key="d_cols")
    if sel:
        r = S2.descriptives(df, group_col, sel)
        if guard(r):
            table(r["table"])
            reg("وصفي", r["table"], r["interp"])
            st.info(r["interp"])
        means_chart([c for c in sel if c in time_cols])
        if len(sel) >= 1:
            st.subheader("افتراضات الاختبارات")
            nm = S2.normality_table(df, group_col, sel)
            if guard(nm):
                table(nm["table"], "اختبار الاعتدالية Shapiro–Wilk لكل مجموعة/قياس")
                reg("اعتدالية", nm["table"], nm["interp"])
                st.info(nm["interp"])
            lev_rows = []
            for c in sel:
                lv = S2.levene_test(groups_of(c))
                if lv.get("ok"):
                    lev_rows.append({"القياس": c, "F": lv["W"], "df1": lv["df1"], "df2": lv["df2"], "p": lv["p"],
                                     "التجانس": "متحقق" if lv["equal"] else "غير متحقق"})
            if lev_rows:
                lt = pd.DataFrame(lev_rows)
                table(lt, "تجانس التباين بين المجموعات (Brown–Forsythe/Levene)")
                reg("ليفين", lt, "اختبار تجانس التباين لكل قياس.")
                st.info(f"تجانس التباين غير متحقق في {int((lt['التجانس'] == 'غير متحقق').sum())} من {len(lt)} قياسات (α = 0.05).")
            ot = S2.outliers_iqr(df, group_col, sel, id_col=id_col)
            if guard(ot):
                table(ot["table"], "القيم الشاذة (قاعدة 1.5×IQR)")
                reg("قيم شاذة", ot["table"], ot["interp"])
                st.info(ot["interp"])
            if len(time_cols) >= 3:
                sp = S2.sphericity(df, time_cols, group_col)
                if guard(sp):
                    table(sp["table"], "الكرويّة (معامل Greenhouse–Geisser)")
                    reg("كروية", sp["table"], sp["interp"])
                    st.info(sp["interp"])
    else:
        st.info("اختر متغيرًا واحدًا على الأقل.")

# ================================================================== 2) مجموعتان
with tabs[1]:
    st.subheader("مقارنة مجموعتين")
    if len(groups_all) < 2:
        st.info("يلزم وجود مجموعتين على الأقل في البيانات.")
    else:
        ca, cb, cc = st.columns(3)
        g1 = ca.selectbox("المجموعة الأولى", groups_all, index=0, key="t2_g1")
        g2 = cb.selectbox("المجموعة الثانية", [g for g in groups_all if g != g1], key="t2_g2")
        col = cc.selectbox("القياس/المتغير", numeric_cols, key="t2_col")
        a, b = groups_of(col, [g1, g2])[g1], groups_of(col, [g1, g2])[g2]
        fam = fam_for([a, b], "المقارنة")
        r = S2.two_independent(a, b, fam, names=(g1, g2))
        if guard(r):
            table(r["table"], "نتيجة الاختبار")
            reg("مجموعتان", r["table"], r["interp"])
            verdict(r)
            st.bar_chart(pd.DataFrame({"المتوسط" if fam == "param" else "الوسيط": [np.mean(a) if fam == "param" else np.median(a),
                                                                                  np.mean(b) if fam == "param" else np.median(b)]},
                                      index=[g1, g2]))

# ================================================================== 3) عدة مجموعات
with tabs[2]:
    st.subheader("مقارنة عدة مجموعات مستقلة")
    if len(groups_all) < 2:
        st.info("يلزم وجود مجموعتين على الأقل.")
    else:
        ca, cb = st.columns(2)
        col = ca.selectbox("القياس/المتغير", numeric_cols, key="t3_col")
        gs = cb.multiselect("المجموعات", groups_all, default=groups_all, key="t3_gs")
        grp = groups_of(col, gs)
        fam = fam_for(list(grp.values()), "المقارنة")
        if fam == "param":
            ph = st.selectbox("المقارنات البعدية", ["auto", "tukey", "games-howell", "holm", "bonferroni"],
                              format_func=lambda x: {"auto": "تلقائي (Tukey أو Games-Howell حسب التجانس)", "tukey": "Tukey HSD",
                                                     "games-howell": "Games–Howell (تباين غير متساوٍ)", "holm": "Welch t + Holm",
                                                     "bonferroni": "Welch t + Bonferroni"}[x], key="t3_ph")
            r = S2.k_independent(grp, "param", posthoc=ph, adjust=ADJUST)
        else:
            r = S2.k_independent(grp, "nonparam", adjust=ADJUST if ADJUST in ("holm", "bonferroni") else "holm")
        if guard(r):
            table(r["table"], "الاختبار الشامل")
            reg("عدة مجموعات", r["table"])
            if r.get("ranks") is not None:
                table(r["ranks"], "الوسيط ومتوسط الرتب")
            table(r.get("posthoc"), "المقارنات البعدية")
            reg("بعدية عدة مجموعات", r.get("posthoc"), r["interp"])
            verdict(r)
            m = pd.DataFrame({"القيمة": [np.mean(v) if fam == "param" else np.median(v) for v in grp.values() if len(v)]},
                             index=[g for g, v in grp.items() if len(v)])
            st.bar_chart(m)

# ================================================================== 4) القياسات المتكررة
with tabs[3]:
    st.subheader("القياسات المتكررة داخل المجموعة")
    if len(time_cols) < 2:
        st.info("يلزم قياسان زمنيان على الأقل (اختر أكثر من عمود قيمة / نقطة قياس).")
    else:
        ca, cb = st.columns(2)
        gsel = ca.selectbox("المجموعة", ["كل اللاعبين معًا"] + groups_all, key="t4_g")
        tsel = cb.multiselect("نقاط القياس", time_cols, default=time_cols, key="t4_t")
        tsel = [c for c in time_cols if c in tsel]
        sub = df if gsel == "كل اللاعبين معًا" else df[df[group_col] == gsel]
        cc_ = sub[tsel].dropna() if tsel else pd.DataFrame()
        if len(tsel) < 2:
            st.info("اختر قياسين على الأقل.")
        elif len(cc_) < 3:
            st.warning(f"عدد الحالات الكاملة ({len(cc_)}) أقل من 3؛ لا يمكن إجراء التحليل.")
        else:
            Y = cc_.to_numpy(float)
            resid = (Y - Y.mean(0) - Y.mean(1)[:, None] + Y.mean()).ravel()
            fam = fam_for(None, "القياسات المتكررة", resid=resid)
            r = S2.rm_oneway(cc_, tsel, fam, adjust=ADJUST)
            if guard(r):
                table(r["means"], "الوصف لكل قياس")
                table(r["table"], "الاختبار الشامل")
                table(r["posthoc"], f"المقارنات الزوجية ({'t مترابط' if fam == 'param' else 'Wilcoxon'} + {adj_label})")
                reg(f"متكررة {gsel}"[:28], r["table"], f"{gsel}: " + r["interp"])
                reg(f"بعدية متكررة {gsel}"[:28], r["posthoc"])
                verdict(r)
                if len(tsel) == 2:
                    pr = S2.two_related(cc_[tsel[0]], cc_[tsel[1]], fam, names=(tsel[0], tsel[1]))
                    if guard(pr):
                        table(pr["table"], "المقارنة الزوجية التفصيلية")
                        verdict(pr)
                traj = cc_.T.copy()
                traj.index = [f"{i + 1}) {c}" for i, c in enumerate(tsel)]
                traj.columns = [f"#{i + 1}" for i in range(traj.shape[1])]
                st.markdown("**مسارات اللاعبين الفردية عبر القياسات**")
                st.line_chart(traj)

# ================================================================== 5) المختلط + ANCOVA
with tabs[4]:
    st.subheader("التصميم المختلط: مجموعة × زمن")
    if len(time_cols) < 2 or len(groups_all) < 2:
        st.info("يلزم قياسان زمنيان ومجموعتان على الأقل.")
    else:
        tsel = st.multiselect("نقاط القياس", time_cols, default=time_cols, key="t5_t")
        tsel = [c for c in time_cols if c in tsel]
        cc_ = wide_clean(tsel) if len(tsel) >= 2 else pd.DataFrame()
        if len(tsel) < 2:
            st.info("اختر قياسين على الأقل.")
        elif len(cc_) < len(groups_all) + 2:
            st.warning(f"عدد الحالات الكاملة ({len(cc_)}) غير كافٍ للتصميم المختلط.")
        else:
            st.caption(f"الحالات الكاملة المستخدمة: {len(cc_)} من {len(df)}.")
            # بواقي النموذج المختلط للتقدير التلقائي
            Yv = cc_[tsel].to_numpy(float)
            gv = cc_[group_col].to_numpy()
            res_ = np.empty_like(Yv)
            for g in pd.unique(gv):
                ix = gv == g
                Yg = Yv[ix]
                res_[ix] = Yg - Yg.mean(0) - Yg.mean(1)[:, None] + Yg.mean()
            fam = fam_for(None, "التصميم المختلط", resid=res_.ravel())
            means_chart(tsel, "متوسطات المجموعات عبر الزمن (الحالات الكاملة)")
            if fam == "param":
                mx = S2.mixed_anova(cc_, group_col, tsel)
                if guard(mx):
                    table(mx["table"], "جدول ANOVA المختلط")
                    table(mx["cells"], "المتوسطات والانحرافات لكل خلية")
                    table(mx["levene"], "تجانس التباين عند كل زمن")
                    reg("مختلط", mx["table"], mx["interp"])
                    reg("مختلط-خلايا", mx["cells"])
                    verdict(mx, mx["p"]["GT"])
                    se = S2.mixed_simple_effects(cc_, group_col, tsel, "param", ADJUST)
                    if guard(se):
                        table(se["between_at_time"], "التأثيرات البسيطة: المجموعات عند كل زمن")
                        table(se["within_group"], "التأثيرات البسيطة: الزمن داخل كل مجموعة")
                        reg("بسيطة-بين", se["between_at_time"])
                        reg("بسيطة-داخل", se["within_group"], se["interp"])
                        st.info(se["interp"])
            else:
                mn = S2.mixed_nonparametric(cc_, group_col, tsel, adjust=ADJUST if ADJUST in ("holm", "bonferroni", "bh") else "holm")
                if guard(mn):
                    st.warning("لا يوجد اختبار لا معلمي دقيق مكافئ لـ ANOVA المختلط؛ يُعرض بديل: Kruskal–Wallis على درجات التغير + Friedman داخل كل مجموعة.")
                    table(mn["kw_change"], "مقارنة المجموعات في درجات التغير (Tk − T0): Kruskal–Wallis")
                    table(mn["friedman"], "التغير الزمني داخل كل مجموعة: Friedman")
                    reg("KW تغير", mn["kw_change"], mn["interp"])
                    reg("Friedman مجموعات", mn["friedman"])
                    for t_, ph_ in mn["dunn"].items():
                        table(ph_, f"Dunn البعدي لدرجة التغير حتى {t_}")
                        reg(f"Dunn {t_}"[:28], ph_)
                    any_sig = any(len(t_) and (t_["القرار"] == "دال").any() for t_ in (mn["kw_change"], mn["friedman"]))
                    (st.success if any_sig else st.info)(mn["interp"])

    st.divider()
    st.subheader("ANCOVA — تحليل التغاير (ضبط القياس القبلي)")
    if len(time_cols) < 2 or len(groups_all) < 2:
        st.info("يلزم قياس قبلي وبعدي ومجموعتان على الأقل.")
    else:
        ca, cb = st.columns(2)
        basec = ca.selectbox("القياس القبلي (المصاحب)", time_cols, index=0, key="t5_base")
        postc = cb.selectbox("القياس البعدي (التابع)", [c for c in time_cols if c != basec], key="t5_post")
        if FORCE == "auto":
            rp = S2.ancova(df, group_col, basec, postc)
            if rp.get("ok") and np.isfinite(rp["resid_shapiro_p"]):
                famA = "param" if rp["resid_shapiro_p"] > .05 else "nonparam"
                st.caption(f"الأسلوب المعتمد لـ ANCOVA: **{'معلمي' if famA == 'param' else 'لا معلمي (ANCOVA على الرتب)'}** — "
                           f"Shapiro–Wilk لبواقي النموذج: p = {rp['resid_shapiro_p']:.3f}.")
            else:
                famA = "nonparam"
                st.caption("تعذّر فحص اعتدالية البواقي؛ اختير ANCOVA على الرتب احتياطًا.")
        else:
            famA = FORCE
        ra = S2.ancova(df, group_col, basec, postc, rank=(famA == "nonparam"), adjust=ADJUST)
        if guard(ra):
            table(ra["table"], "جدول ANCOVA")
            table(ra["adjusted"], "المتوسطات الخام والمعدَّلة")
            table(ra["pairwise"], "مقارنات المتوسطات المعدَّلة")
            reg("ANCOVA", ra["table"], ra["interp"])
            reg("ANCOVA-معدلة", ra["adjusted"])
            reg("ANCOVA-مقارنات", ra["pairwise"])
            verdict(ra)
            if ra["resid_levene"].get("ok"):
                st.caption("تجانس تباين البواقي: " + ra["resid_levene"]["interp"])
            adj_col = [c for c in ra["adjusted"].columns if "المعدَّل" in c][0]
            st.bar_chart(ra["adjusted"].set_index("المجموعة")[adj_col])

# ================================================================== 6) الارتباط والانحدار
with tabs[5]:
    st.subheader("الارتباط")
    cols_sel = st.multiselect("المتغيرات", numeric_cols, default=time_cols[:min(len(time_cols), 4)], key="t6_cols")
    METH = {"بيرسون (معلمي)": "pearson", "سبيرمان (لا معلمي)": "spearman", "كيندال تاو-b (لا معلمي)": "kendall"}
    if len(cols_sel) < 2:
        st.info("اختر متغيرين على الأقل.")
    else:
        if FORCE == "param":
            default_m = "pearson"
        elif FORCE == "nonparam":
            default_m = "spearman"
        else:
            nn = S2.normality_table(df, group_col, cols_sel)
            all_norm = bool(nn.get("all_normal")) if nn.get("ok") else False
            default_m = "pearson" if all_norm else "spearman"
            st.caption(f"الاختيار التلقائي: {'بيرسون (الاعتدالية متحققة)' if all_norm else 'سبيرمان (الاعتدالية غير متحققة في بعض المتغيرات)'}.")
        m_label = st.selectbox("معامل الارتباط", list(METH), index=list(METH.values()).index(default_m), key="t6_m")
        cm = S2.correlation_matrix(df, cols_sel, METH[m_label], adjust=ADJUST)
        if guard(cm):
            table(cm["r"].reset_index().rename(columns={"index": ""}), "مصفوفة المعاملات")
            table(cm["p"].reset_index().rename(columns={"index": ""}), f"مصفوفة قيم p ({adj_label})")
            table(cm["pairs"], "الأزواج مع حدود الثقة 95%")
            reg("ارتباط-مصفوفة", cm["r"].reset_index())
            reg("ارتباط-أزواج", cm["pairs"], cm["interp"])
            st.info(cm["interp"])
        xa, ya = st.columns(2)
        xv = xa.selectbox("محور X", cols_sel, index=0, key="t6_x")
        yv = ya.selectbox("محور Y", [c for c in cols_sel if c != xv], key="t6_y")
        pair = df[[xv, yv, group_col]].dropna()
        if len(pair) >= 4:
            st.scatter_chart(pair, x=xv, y=yv, color=group_col)
            cp = S2.correlation_pair(pair[xv], pair[yv], METH[m_label], names=(xv, yv))
            if guard(cp):
                verdict(cp)
    st.divider()
    st.subheader("الانحدار الخطي (بسيط/متعدد)")
    ya_, xa_ = st.columns(2)
    ycol = ya_.selectbox("المتغير التابع (Y)", numeric_cols, index=min(1, len(numeric_cols) - 1), key="t6_ry")
    xcols = xa_.multiselect("المتغيرات المستقلة (X)", [c for c in numeric_cols if c != ycol],
                            default=[c for c in numeric_cols if c != ycol][:1], key="t6_rx")
    if xcols:
        lr = S2.linear_regression(df, ycol, xcols)
        if guard(lr):
            table(lr["model"], "ملخص النموذج")
            table(lr["coef"], "المعاملات")
            reg("انحدار-نموذج", lr["model"], lr["interp"])
            reg("انحدار-معاملات", lr["coef"])
            verdict(lr)
            fit = pd.DataFrame({"القيم المتنبأ بها": lr["fitted"], "البواقي": lr["resid"]})
            st.markdown("**البواقي مقابل القيم المتنبأ بها**")
            st.scatter_chart(fit, x="القيم المتنبأ بها", y="البواقي")
    else:
        st.info("اختر متغيرًا مستقلًا واحدًا على الأقل.")

# ================================================================== 7) الجداول التكرارية
with tabs[6]:
    st.subheader("المستجيبون (Responders)")
    if len(time_cols) < 2:
        st.info("يلزم قياس قبلي وبعدي.")
    else:
        ca, cb, cc = st.columns(3)
        b_ = ca.selectbox("القبلي", time_cols, index=0, key="t7_b")
        p_ = cb.selectbox("البعدي", [c for c in time_cols if c != b_], key="t7_p")
        direction = cc.radio("اتجاه التحسن", ["انخفاض (مثل الألم)", "ارتفاع (مثل المدى/القوة)"], key="t7_d")
        cd, ce = st.columns(2)
        md = cd.number_input("الحد الأدنى المطلق للتحسن", min_value=0.0, value=2.0, step=0.5, key="t7_md")
        mp = ce.number_input("الحد الأدنى النسبي للتحسن (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0, key="t7_mp")
        rr = S2.responders_analysis(df, group_col, b_, p_, md, mp / 100.0, "decrease" if direction.startswith("انخفاض") else "increase")
        if guard(rr):
            table(rr["table"], "نسب المستجيبين")
            reg("مستجيبون", rr["table"], rr["interp"])
            verdict(rr)
            st.bar_chart(rr["table"].set_index("المجموعة")["النسبة %"])
            if rr.get("chi"):
                table(rr["chi"]["table"], "تفاصيل الاختبار")
                table(rr["chi"]["observed"].reset_index(), "التكرارات المشاهَدة")
                table(rr["chi"]["expected"].reset_index(), "التكرارات المتوقعة")
    st.divider()
    st.subheader("جدول تكراري عام (χ² / فيشر)")
    cat_cols = [c for c in df.columns if (df[c].dtype == object or df[c].nunique() <= 6) and c not in (id_col,)]
    ca, cb = st.columns(2)
    rowv = ca.selectbox("المتغير الأول (صفوف)", [group_col] + [c for c in cat_cols if c != group_col], key="t7_r")
    colv = cb.selectbox("المتغير الثاني (أعمدة)", [c for c in cat_cols if c != rowv] + numeric_cols, key="t7_c")
    work = df.copy()
    if pd.api.types.is_numeric_dtype(work[colv]) and work[colv].nunique() > 6:
        cut = st.number_input(f"قيمة القطع لتحويل «{colv}» إلى فئتين (≥ القيمة = مرتفع)", value=float(np.nanmedian(work[colv])), key="t7_cut")
        work[colv] = work[colv].map(lambda v: np.nan if pd.isna(v) else (f"≥ {cut:g}" if v >= cut else f"< {cut:g}"))
    mth = st.radio("الاختبار", ["تلقائي", "χ²", "فيشر الدقيق (2×2)"], horizontal=True, key="t7_m")
    ct = S2.crosstab_test(work, rowv, colv, method={"تلقائي": "auto", "χ²": "chi2", "فيشر الدقيق (2×2)": "fisher"}[mth])
    if guard(ct):
        table(ct["observed"].reset_index(), "التكرارات المشاهَدة")
        table(ct["row_pct"].round(1).reset_index(), "النسب المئوية داخل الصفوف (%)")
        table(ct["expected"].reset_index(), "التكرارات المتوقعة")
        table(ct["table"], "نتائج الاختبار")
        reg("تكراري-مشاهد", ct["observed"].reset_index())
        reg("تكراري-اختبار", ct["table"], ct["interp"])
        verdict(ct)

# ================================================================== 8) حاسبة القدرة
with tabs[7]:
    st.subheader("حاسبة حجم العينة والقدرة الإحصائية")
    kind = st.radio("نوع التحليل", ["t لمجموعتين مستقلتين", "t مترابط (قبلي/بعدي)", "ANOVA أحادي الاتجاه"], horizontal=True, key="t8_k")
    K = {"t لمجموعتين مستقلتين": "two", "t مترابط (قبلي/بعدي)": "paired", "ANOVA أحادي الاتجاه": "anova"}[kind]
    ca, cb, cc = st.columns(3)
    if K == "anova":
        eff = ca.number_input("حجم الأثر f (كوهين: 0.10 صغير، 0.25 متوسط، 0.40 كبير)", min_value=0.01, max_value=2.0, value=0.25, step=0.05, key="t8_e")
        kk = int(cb.number_input("عدد المجموعات", min_value=2, max_value=10, value=3, step=1, key="t8_kk"))
    else:
        eff = ca.number_input("حجم الأثر " + ("d" if K == "two" else "d_z") + " (0.2 صغير، 0.5 متوسط، 0.8 كبير)", min_value=0.05, max_value=3.0, value=0.5, step=0.05, key="t8_e")
        kk = 3
    pw_t = cc.number_input("القدرة المطلوبة", min_value=0.5, max_value=0.99, value=0.80, step=0.05, key="t8_p")
    nin = int(st.number_input("حجم العينة لكل مجموعة (أو عدد الأزواج) لحساب القدرة", min_value=3, max_value=2000, value=15, step=1, key="t8_n"))
    alpha_p = 0.05
    req = {"two": lambda: S2.n_required_two_sample(eff, pw_t, alpha_p), "paired": lambda: S2.n_required_paired(eff, pw_t, alpha_p),
           "anova": lambda: S2.n_required_anova(eff, kk, pw_t, alpha_p)}[K]()
    st.success(S2.power_summary(K, eff, None, kk, pw_t, alpha_p))
    st.info(S2.power_summary(K, eff, nin, kk, pw_t, alpha_p))
    ns = list(range(max(3, nin // 3), max(60, (req or 60) * 2) + 1, 2))
    fn_ = {"two": lambda n: S2.power_two_sample(eff, n, alpha_p), "paired": lambda n: S2.power_paired(eff, n, alpha_p),
           "anova": lambda n: S2.power_anova(eff, kk, n, alpha_p)}[K]
    curve = pd.DataFrame({"القدرة": [fn_(n) for n in ns], "0.80": [0.8] * len(ns)}, index=ns)
    st.markdown("**منحنى القدرة مقابل حجم العينة**")
    st.line_chart(curve)
    st.caption("الحسابات بالتوزيعات اللامركزية (t وF) لاختبار ثنائي الطرف وα = 0.05؛ لتصاميم القياسات المتكررة/المختلطة تقريبية فقط، "
               "ويُنصح بـ G*Power لهذه التصاميم.")

# ------------------------------------------------------------------ التصدير
st.divider()
st.subheader("تصدير النتائج")
try:
    xlsx = S2.results_to_excel(RES_TABLES, RES_NOTES)
    st.download_button("⬇️ كل جداول النتائج (Excel)", xlsx, "نتائج_الاحصاء.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as e:  # noqa: BLE001
    st.warning(f"تعذّر إنشاء ملف النتائج: {e}")
st.download_button("⬇️ البيانات المحلَّلة (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "البيانات_المحللة.csv", "text/csv")
st.caption("ملاحظة منهجية: لا تحل هذه الصفحة محل استشارة إحصائي عند النماذج الأعقد (LMM، تعويض القيم المفقودة)، "
           "لكنها تغطي الاختبارات المعتادة في أطروحات التربية الرياضية.")
