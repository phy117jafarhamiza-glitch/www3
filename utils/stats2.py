# -*- coding: utf-8 -*-
"""
stats2 — وحدة إحصائية شاملة (بديل SPSS) : معلمية ولا معلمية، دوال نقية بلا streamlit.

كل دالة اختبار تعيد قاموسًا فيه  ok  وجداول بتسميات عربية ونص تفسير عربي  interp
يذكر الاختبار والإحصاءة ودرجات الحرية وقيمة p وحجم الأثر والقرار عند α = 0.05.
عند عدم كفاية البيانات يعاد  {"ok": False, "reason": "..."}  ولا يُرفع استثناء.

ملاحظات منهجية:
 - لا يوجد ANOVA مختلط لا معلمي دقيق؛ يُقدَّم بديل وصفي-استدلالي (Kruskal على درجات التغير
   + Friedman داخل كل مجموعة) ويُنبَّه إلى ذلك في التفسير.
 - في ANOVA المختلط بعدد غير متساوٍ تُستخدم مجاميع المربعات «الموزونة» (النوع I)؛
   وهي مطابقة للنوع III في التصميم المتوازن.
"""
import functools
import io
import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# =============================================================== مساعدات عامة
def _fail(reason):
    return {"ok": False, "reason": reason}


def _safe(fn):
    """يمنع انهيار الواجهة: أي استثناء غير متوقع يتحول إلى رسالة عربية."""
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            return _fail(f"تعذّر إجراء التحليل لهذه البيانات ({type(e).__name__}: {e}).")
    return wrap


def _clean(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def fn(x, d=2):
    try:
        if x is None or not np.isfinite(x):
            return "—"
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "—"


def fp(p):
    if p is None or not np.isfinite(p):
        return "p = —"
    if p < .001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0") if p < 1 else "p = 1.000"


def mag(v, kind="d"):
    th = {"d": (.2, .5, .8), "eta": (.01, .06, .14), "r": (.1, .3, .5),
          "w": (.1, .3, .5), "eps": (.01, .08, .26)}[kind]
    a = abs(v) if v is not None and np.isfinite(v) else 0.0
    return "ضئيل" if a < th[0] else "صغير" if a < th[1] else "متوسط" if a < th[2] else "كبير"


def _sig(p, alpha=ALPHA):
    return bool(p is not None and np.isfinite(p) and p < alpha)


def _verdict(p, what_sig, what_ns, alpha=ALPHA):
    if _sig(p, alpha):
        return f"القرار عند α = {alpha:g}: {what_sig}."
    return (f"القرار عند α = {alpha:g}: {what_ns}؛ وعدم الدلالة لا يعني إثبات التساوي "
            "(قد تكون القدرة الإحصائية محدودة بصغر العينة).")


def p_adjust(pvals, method="holm"):
    """تصحيح المقارنات المتعددة: holm | bonferroni | bh | none."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    if method in (None, "none"):
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(1.0, p * n).tolist()
    order = np.argsort(p)
    adj = np.empty(n)
    if method == "bh":
        running = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            running = min(running, p[idx] * n / (rank + 1))
            adj[idx] = running
        return adj.tolist()
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


ADJ_AR = {"holm": "Holm", "bonferroni": "Bonferroni", "bh": "Benjamini–Hochberg", "none": "بدون تصحيح"}


def _tie_term(values):
    """مجموع (t³ − t) لمجموعات القيم المتساوية."""
    _, c = np.unique(values, return_counts=True)
    return float(((c ** 3) - c).sum())


# =============================================================== الوصفي
@_safe
def descriptives(wide, group_col, cols, overall=True):
    """جدول وصفي لكل مجموعة × متغير/زمن."""
    rows = []
    sets = [(g, sub) for g, sub in wide.groupby(group_col, sort=False)]
    if overall:
        sets.append(("الكل", wide))
    for g, sub in sets:
        for c in cols:
            raw = pd.to_numeric(sub[c], errors="coerce")
            v = raw.dropna().to_numpy(float)
            n = len(v)
            if n == 0:
                continue
            m = v.mean()
            sd = v.std(ddof=1) if n > 1 else np.nan
            sem = sd / np.sqrt(n) if n > 1 else np.nan
            if n > 1:
                tc = stats.t.ppf(.975, n - 1)
                lo, hi = m - tc * sem, m + tc * sem
            else:
                lo = hi = np.nan
            q1, q3 = np.percentile(v, [25, 75])
            rows.append({
                "المجموعة": g, "المتغير/القياس": c, "n": n, "مفقودة": int(raw.isna().sum()),
                "المتوسط": m, "الانحراف المعياري": sd, "الخطأ المعياري": sem,
                "95% CI أدنى": lo, "95% CI أعلى": hi,
                "الوسيط": float(np.median(v)), "Q1": q1, "Q3": q3, "المدى الربيعي IQR": q3 - q1,
                "الأدنى": v.min(), "الأعلى": v.max(),
                "الالتواء": float(stats.skew(v, bias=False)) if n > 2 and np.ptp(v) > 0 else np.nan,
                "التفلطح": float(stats.kurtosis(v, bias=False)) if n > 3 and np.ptp(v) > 0 else np.nan,
            })
    if not rows:
        return _fail("لا توجد قيم رقمية كافية للوصف.")
    return {"ok": True, "table": pd.DataFrame(rows),
            "interp": "الجدول يعرض المتوسط والانحراف المعياري والوسيط والمدى الربيعي وحدود الثقة 95% لكل مجموعة وقياس؛ "
                      "عند وجود التواء واضح أو قيم شاذة يُفضَّل الاعتماد على الوسيط والمدى الربيعي."}


@_safe
def normality_table(wide, group_col, cols, alpha=ALPHA):
    """Shapiro–Wilk لكل مجموعة وقياس."""
    rows = []
    for g, sub in wide.groupby(group_col, sort=False):
        for c in cols:
            v = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy(float)
            if len(v) >= 3 and np.ptp(v) > 0:
                w, p = stats.shapiro(v)
                rows.append({"المجموعة": g, "القياس": c, "n": len(v), "W": float(w), "p": float(p),
                             "التوزيع": "طبيعي (لا يُرفض)" if p > alpha else "غير طبيعي"})
            else:
                rows.append({"المجموعة": g, "القياس": c, "n": len(v), "W": np.nan, "p": np.nan,
                             "التوزيع": "غير قابل للاختبار (n<3 أو قيم متطابقة)"})
    if not rows:
        return _fail("لا توجد بيانات.")
    t = pd.DataFrame(rows)
    ok = t["p"].dropna()
    nviol = int((ok <= alpha).sum())
    interp = (f"اختبار Shapiro–Wilk على {len(ok)} مجموعة/قياس: رُفضت الاعتدالية في {nviol} منها عند α = {alpha:g}. "
              "مع العينات الصغيرة (n≈15) يكون الاختبار ضعيف القدرة على كشف الانحراف، لذا يُستحسن دعمه بالرسوم "
              "وبمؤشرات الالتواء والتفلطح.")
    return {"ok": True, "table": t, "interp": interp, "all_normal": nviol == 0 and len(ok) > 0}


@_safe
def levene_test(groups, center="median", alpha=ALPHA):
    """اختبار ليفين (center='median' = Brown–Forsythe)."""
    arrs = {k: _clean(v) for k, v in groups.items()}
    arrs = {k: v for k, v in arrs.items() if len(v) >= 2}
    if len(arrs) < 2:
        return _fail("يلزم مجموعتان على الأقل (n≥2) لاختبار تجانس التباين.")
    if all(np.ptp(v) == 0 for v in arrs.values()):
        return _fail("جميع القيم متطابقة داخل المجموعات.")
    W, p = stats.levene(*arrs.values(), center=center)
    k, N = len(arrs), sum(len(v) for v in arrs.values())
    name = "Brown–Forsythe (ليفين حول الوسيط)" if center == "median" else "Levene (حول المتوسط)"
    return {"ok": True, "W": float(W), "df1": k - 1, "df2": N - k, "p": float(p), "equal": bool(p > alpha),
            "interp": (f"اختبار {name}: F({k - 1}, {N - k}) = {fn(W)}, {fp(p)}؛ "
                       + ("لا يوجد دليل على اختلاف التباينات (يُقبل افتراض التجانس)." if p > alpha
                          else "تباينات المجموعات غير متجانسة؛ يُفضَّل Welch/Games-Howell أو اختبار لا معلمي."))}


@_safe
def outliers_iqr(wide, group_col, cols, k=1.5, id_col=None):
    """القيم الشاذة بقاعدة IQR (Tukey): خارج [Q1−k·IQR, Q3+k·IQR]؛ متطرفة إن تجاوزت 3·IQR."""
    rows = []
    for g, sub in wide.groupby(group_col, sort=False):
        for c in cols:
            s = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(s) < 4:
                continue
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            lo3, hi3 = q1 - 3 * iqr, q3 + 3 * iqr
            for idx, val in s.items():
                if val < lo or val > hi:
                    rows.append({"المجموعة": g, "القياس": c,
                                 "المعرّف": (sub.loc[idx, id_col] if id_col and id_col in sub.columns else idx),
                                 "القيمة": float(val), "الحد الأدنى": lo, "الحد الأعلى": hi,
                                 "النوع": "متطرفة (>3×IQR)" if (val < lo3 or val > hi3) else "شاذة"})
    t = pd.DataFrame(rows, columns=["المجموعة", "القياس", "المعرّف", "القيمة", "الحد الأدنى", "الحد الأعلى", "النوع"])
    interp = ("لم تُرصد قيم شاذة وفق قاعدة IQR." if len(t) == 0 else
              f"رُصدت {len(t)} قيمة شاذة وفق قاعدة {k:g}×IQR؛ تحقق من صحة إدخالها قبل استبعادها، "
              "ولا تُحذف إلا بمبرر منهجي موثَّق.")
    return {"ok": True, "table": t, "interp": interp}


def gg_epsilon(S):
    """معامل Greenhouse–Geisser (و Huynh–Feldt) من مصفوفة التباين المشترك k×k. يعيد (gg, lower_bound)."""
    S = np.asarray(S, float)
    k = S.shape[0]
    if k < 3:
        return 1.0, 1.0
    C = np.eye(k) - np.ones((k, k)) / k
    Sc = C @ S @ C
    num = np.trace(Sc) ** 2
    den = (k - 1) * np.trace(Sc @ Sc)
    eps = float(num / den) if den > 0 else 1.0
    return float(min(1.0, max(1.0 / (k - 1), eps))), 1.0 / (k - 1)


def hf_epsilon(eps_gg, n, k, n_groups=1):
    """Huynh–Feldt (مصحَّح ≤ 1)."""
    if k < 3:
        return 1.0
    N = n
    num = N * (k - 1) * eps_gg - 2
    den = (k - 1) * (N - n_groups - (k - 1) * eps_gg)
    return float(min(1.0, num / den)) if den > 0 else 1.0


@_safe
def sphericity(wide, time_cols, group_col=None):
    """تقرير الكرويّة بدون Mauchly: ε (Greenhouse–Geisser) وHuynh–Feldt والحد الأدنى."""
    cols = list(time_cols)
    cc = wide[([group_col] if group_col else []) + cols].dropna()
    k = len(cols)
    if k < 3:
        return _fail("الكرويّة تُفحص عند وجود 3 قياسات فأكثر (مع قياسين ε = 1 تلقائيًا).")
    if group_col:
        a = cc[group_col].nunique()
        N = len(cc)
        if N - a < k:
            return _fail("عدد المفحوصين غير كافٍ لتقدير مصفوفة التباين المشترك.")
        S = np.zeros((k, k))
        for _, sub in cc.groupby(group_col):
            Y = sub[cols].to_numpy(float)
            Yc = Y - Y.mean(0)
            S += Yc.T @ Yc
        S /= (N - a)
        n_eff, ng = N, a
    else:
        Y = cc[cols].to_numpy(float)
        if len(Y) < k + 1:
            return _fail("عدد المفحوصين غير كافٍ لتقدير مصفوفة التباين المشترك.")
        S = np.cov(Y, rowvar=False)
        n_eff, ng = len(Y), 1
    gg, lb = gg_epsilon(S)
    hf = hf_epsilon(gg, n_eff, k, ng)
    t = pd.DataFrame([{"ε (Greenhouse–Geisser)": gg, "ε (Huynh–Feldt)": hf, "الحد الأدنى 1/(k−1)": lb,
                       "عدد القياسات k": k, "عدد الحالات الكاملة": n_eff}])
    interp = (f"معامل Greenhouse–Geisser ε = {fn(gg, 3)} (Huynh–Feldt = {fn(hf, 3)}). "
              + ("القيمة قريبة من 1 فالكرويّة معقولة." if gg >= .90 else
                 "القيمة أقل من 0.90 أي أن افتراض الكرويّة منتهك؛ يُعتمد تصحيح Greenhouse–Geisser لدرجات الحرية."))
    return {"ok": True, "table": t, "eps_gg": gg, "eps_hf": hf, "interp": interp}


# =============================================================== اختيار الأسرة
def choose_family(data_groups, force="auto", alpha=ALPHA):
    """يقرر (family, reason) حيث family ∈ {'param','nonparam'}.
    data_groups: dict أو list لمصفوفات المجموعات. auto: Shapiro لكل مجموعة (p>.05 و n≥3) + ليفين."""
    if isinstance(data_groups, dict):
        arrs = [_clean(v) for v in data_groups.values()]
    else:
        arrs = [_clean(v) for v in data_groups]
    if force == "param":
        return "param", "اختير الاختبار المعلمي يدويًا."
    if force == "nonparam":
        return "nonparam", "اختير الاختبار اللامعلمي يدويًا."
    arrs = [a for a in arrs if len(a) > 0]
    if not arrs:
        return "nonparam", "لا بيانات كافية؛ اختير اللامعلمي احتياطًا."
    small = [len(a) for a in arrs if len(a) < 3]
    if small:
        return "nonparam", "إحدى المجموعات n<3 فلا يمكن فحص الاعتدالية؛ اختير اللامعلمي احتياطًا."
    bad = []
    for i, a in enumerate(arrs):
        if np.ptp(a) == 0:
            bad.append(i)
            continue
        if stats.shapiro(a)[1] <= alpha:
            bad.append(i)
    if bad:
        return "nonparam", (f"رُفضت الاعتدالية (Shapiro–Wilk، p ≤ {alpha:g}) في {len(bad)} من {len(arrs)} "
                            "مجموعة/قياس؛ اختير الاختبار اللامعلمي.")
    reason = f"الاعتدالية مقبولة في كل المجموعات (Shapiro–Wilk، p > {alpha:g}، n≥3)"
    if len(arrs) >= 2 and all(len(a) >= 2 for a in arrs):
        try:
            lp = stats.levene(*arrs, center="median")[1]
            reason += (f"؛ وتجانس التباين مقبول (Levene، {fp(lp)})" if lp > alpha else
                       f"؛ لكن التباينات غير متجانسة (Levene، {fp(lp)}) فتُستخدم صيغ Welch/Games-Howell")
        except Exception:  # noqa: BLE001
            pass
    return "param", reason + "؛ اختير الاختبار المعلمي."


def choose_family_residuals(resid, force="auto", alpha=ALPHA):
    """نسخة للتصاميم المترابطة/المختلطة: اعتدالية البواقي (أو فروق الأزواج)."""
    if force in ("param", "nonparam"):
        return choose_family([resid], force)
    r = _clean(resid)
    if len(r) < 3 or np.ptp(r) == 0:
        return "nonparam", "عدد الحالات قليل لفحص اعتدالية البواقي؛ اختير اللامعلمي احتياطًا."
    p = stats.shapiro(r)[1]
    if p > alpha:
        return "param", f"بواقي/فروق النموذج تقبل الاعتدالية (Shapiro–Wilk، {fp(p)})؛ اختير المعلمي."
    return "nonparam", f"بواقي/فروق النموذج غير طبيعية (Shapiro–Wilk، {fp(p)})؛ اختير اللامعلمي."


# =============================================================== مجموعتان مستقلتان
def hedges_g(a, b):
    """Hedges g مع 95% CI (تقريب Hedges–Olkin)."""
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * J
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2 - 2)))
    return float(g), float(g - 1.959964 * se), float(g + 1.959964 * se)


def mann_whitney_detail(a, b):
    """يعيد dict: U1,U2,U(min),z (بتصحيح الروابط), p, r_rb, r=z/√N, HL."""
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    U1 = float(res.statistic)
    U2 = n1 * n2 - U1
    N = n1 + n2
    allv = np.concatenate([a, b])
    tt = _tie_term(allv)
    sigma = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - tt / (N * (N - 1)))) if N > 1 else np.nan
    z = (U1 - n1 * n2 / 2.0) / sigma if sigma and sigma > 0 else np.nan
    rrb = 2 * U1 / (n1 * n2) - 1
    hl = float(np.median((a[:, None] - b[None, :]).ravel()))
    return {"U1": U1, "U2": U2, "U": min(U1, U2), "z": float(z), "p": float(res.pvalue),
            "r_rb": float(rrb), "r": float(abs(z) / np.sqrt(N)) if np.isfinite(z) else np.nan, "HL": hl,
            "n1": n1, "n2": n2}


def _desc_str(x):
    return f"م = {fn(x.mean())}, ع = {fn(x.std(ddof=1)) if len(x) > 1 else '—'}, ن = {len(x)}"


@_safe
def two_independent(a, b, family="param", names=("المجموعة 1", "المجموعة 2"), alpha=ALPHA, var="auto"):
    """مقارنة مجموعتين مستقلتين. family='param': Student/Welch (حسب ليفين إن var='auto')؛ 'nonparam': Mann–Whitney."""
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return _fail("يلزم لاعبان (قيمتان) على الأقل في كل مجموعة.")
    if np.ptp(np.concatenate([a, b])) == 0:
        return _fail("جميع القيم متطابقة؛ لا يمكن إجراء الاختبار.")
    n_a, n_b = names
    if family == "param":
        m1, m2 = a.mean(), b.mean()
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        lev = levene_test({n_a: a, n_b: b}, alpha=alpha)
        lev_p = lev["p"] if lev.get("ok") else np.nan
        equal = (lev_p > alpha) if var == "auto" and np.isfinite(lev_p) else (var == "equal")
        diff = m1 - m2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se_s = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df_s = n1 + n2 - 2
        se_w = np.sqrt(v1 / n1 + v2 / n2)
        df_w = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        rows = []
        for nm, se, df, flag in (("Student t (تباين متساوٍ)", se_s, df_s, equal), ("Welch t (تباين غير متساوٍ)", se_w, df_w, not equal)):
            t = diff / se
            p = 2 * stats.t.sf(abs(t), df)
            tc = stats.t.ppf(1 - alpha / 2, df)
            rows.append({"الاختبار": nm, "t": t, "درجات الحرية": df, "p": p, "فرق المتوسطات": diff,
                         "الخطأ المعياري للفرق": se, "CI أدنى": diff - tc * se, "CI أعلى": diff + tc * se,
                         "المعتمد": "✔" if flag else ""})
        main = rows[0] if equal else rows[1]
        g, glo, ghi = hedges_g(a, b)
        meth = "Student (تجانس التباين مقبول)" if equal else "Welch (لعدم تجانس التباين)"
        interp = (f"اختبار t لعينتين مستقلتين — {meth}. {n_a} ({_desc_str(a)}) مقابل {n_b} ({_desc_str(b)}). "
                  f"t({fn(main['درجات الحرية'], 1 if not equal else 0)}) = {fn(main['t'])}, {fp(main['p'])}; "
                  f"فرق المتوسطات = {fn(diff)} (95% CI [{fn(main['CI أدنى'])}, {fn(main['CI أعلى'])}])؛ "
                  f"Hedges g = {fn(g)} [{fn(glo)}, {fn(ghi)}] (أثر {mag(g, 'd')}). "
                  + (lev["interp"] + " " if lev.get("ok") else "")
                  + _verdict(main["p"], f"يوجد فرق دال إحصائيًا بين متوسطي {n_a} و{n_b}",
                             f"لا يوجد فرق دال إحصائيًا بين متوسطي {n_a} و{n_b}", alpha))
        return {"ok": True, "family": "param", "test": "Welch t" if not equal else "Student t",
                "stat": float(main["t"]), "df": float(main["درجات الحرية"]), "p": float(main["p"]),
                "effect": {"Hedges g": g, "g CI": (glo, ghi)}, "mean_diff": float(diff),
                "table": pd.DataFrame(rows), "levene": lev, "interp": interp}
    # لا معلمي
    mw = mann_whitney_detail(a, b)
    ra = stats.rankdata(np.concatenate([a, b]))
    tbl = pd.DataFrame([{
        "الاختبار": "Mann–Whitney U", "U": mw["U"], "Z": mw["z"], "p": mw["p"],
        "رتبة المتوسط (مج1)": ra[:n1].mean(), "رتبة المتوسط (مج2)": ra[n1:].mean(),
        "rank-biserial r": mw["r_rb"], "r = |Z|/√N": mw["r"], "إزاحة Hodges–Lehmann": mw["HL"]}])
    interp = (f"اختبار Mann–Whitney U. {n_a}: الوسيط {fn(np.median(a))} (ن = {n1})، {n_b}: الوسيط {fn(np.median(b))} (ن = {n2}). "
              f"U = {fn(mw['U'], 1)}, Z = {fn(mw['z'])}, {fp(mw['p'])}؛ rank-biserial r = {fn(mw['r_rb'])} "
              f"(أثر {mag(mw['r_rb'], 'r')}; الإشارة الموجبة تعني ميل {n_a} لقيم أكبر)، "
              f"إزاحة Hodges–Lehmann = {fn(mw['HL'])}. "
              + _verdict(mw["p"], f"يوجد فرق دال إحصائيًا في توزيع/رتب {n_a} و{n_b}",
                         f"لا يوجد فرق دال إحصائيًا في رتب {n_a} و{n_b}", alpha))
    return {"ok": True, "family": "nonparam", "test": "Mann–Whitney U", "stat": mw["U"], "z": mw["z"], "p": mw["p"],
            "effect": {"rank-biserial": mw["r_rb"], "r": mw["r"]}, "table": tbl, "interp": interp}


# =============================================================== عينتان مترابطتان
@_safe
def two_related(x, y, family="param", names=("القياس 1", "القياس 2"), alpha=ALPHA):
    """x ثم y لنفس الأفراد؛ التغير = y − x. param: paired t ؛ nonparam: Wilcoxon signed-rank."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return _fail("يلزم 3 أزواج كاملة على الأقل.")
    d = y - x
    n_x, n_y = names
    if np.ptp(d) == 0 and d[0] == 0:
        return _fail("لا توجد فروق بين القياسين (كل الفروق صفر).")
    if family == "param":
        sd = d.std(ddof=1)
        if sd == 0:
            return _fail("الفروق متطابقة تمامًا (تباين صفري)؛ لا يمكن حساب t.")
        md = d.mean()
        se = sd / np.sqrt(n)
        t = md / se
        df = n - 1
        p = float(2 * stats.t.sf(abs(t), df))
        tc = stats.t.ppf(1 - alpha / 2, df)
        dz = md / sd
        sed = np.sqrt(1 / n + dz ** 2 / (2 * n))
        dlo, dhi = dz - 1.959964 * sed, dz + 1.959964 * sed
        tbl = pd.DataFrame([{"الاختبار": "t للعينات المترابطة", "n": n, "متوسط التغير": md, "ع التغير": sd,
                             "الخطأ المعياري": se, "t": t, "درجات الحرية": df, "p": p,
                             "CI أدنى": md - tc * se, "CI أعلى": md + tc * se, "d_z": dz,
                             "d_z CI أدنى": dlo, "d_z CI أعلى": dhi}])
        interp = (f"اختبار t للعينات المترابطة ({n_x} → {n_y}، ن = {n} زوجًا): متوسط التغير = {fn(md)} "
                  f"(ع = {fn(sd)}; 95% CI [{fn(md - tc * se)}, {fn(md + tc * se)}])، t({df}) = {fn(t)}, {fp(p)}؛ "
                  f"d_z = {fn(dz)} [{fn(dlo)}, {fn(dhi)}] (أثر {mag(dz, 'd')}). "
                  + _verdict(p, f"يوجد تغيّر دال إحصائيًا بين {n_x} و{n_y}", f"لا يوجد تغيّر دال إحصائيًا بين {n_x} و{n_y}", alpha))
        return {"ok": True, "family": "param", "test": "paired t", "stat": float(t), "df": df, "p": p,
                "effect": {"d_z": float(dz)}, "mean_change": float(md), "table": tbl, "interp": interp}
    # Wilcoxon
    nz = d[d != 0]
    nzero = n - len(nz)
    if len(nz) < 1:
        return _fail("جميع الفروق صفر.")
    res = stats.wilcoxon(nz, alternative="two-sided")
    p = float(res.pvalue)
    ranks = stats.rankdata(np.abs(nz))
    Tp = float(ranks[nz > 0].sum())
    Tm = float(ranks[nz < 0].sum())
    ne = len(nz)
    mu = ne * (ne + 1) / 4
    sig2 = ne * (ne + 1) * (2 * ne + 1) / 24 - _tie_term(np.abs(nz)) / 48
    z = (min(Tp, Tm) - mu) / np.sqrt(sig2) if sig2 > 0 else np.nan
    zsign = -abs(z) if Tp >= Tm else abs(z)  # SPSS: Z سالب عندما الرتب الموجبة أكثر
    r = abs(z) / np.sqrt(n) if np.isfinite(z) else np.nan
    rrb = (Tp - Tm) / (Tp + Tm) if (Tp + Tm) > 0 else np.nan
    tbl = pd.DataFrame([{"الاختبار": "Wilcoxon signed-rank", "n (أزواج)": n, "أزواج صفرية": nzero,
                         "رتب موجبة (ΣR+)": Tp, "رتب سالبة (ΣR−)": Tm, "W": float(res.statistic),
                         "Z": abs(z) if np.isfinite(z) else np.nan, "p": p, "r = |Z|/√n": r,
                         "matched-pairs rank-biserial": rrb, "وسيط التغير": float(np.median(d))}])
    interp = (f"اختبار Wilcoxon للرتب الإشارية ({n_x} → {n_y}، ن = {n} زوجًا"
              + (f"، منها {nzero} فرقًا صفريًا استُبعدت" if nzero else "") + f"): وسيط التغير = {fn(np.median(d))}، "
              f"W = {fn(res.statistic, 1)}, Z = {fn(abs(z) if np.isfinite(z) else np.nan)}, {fp(p)}؛ "
              f"r = Z/√n = {fn(r)} (أثر {mag(r, 'r')})، rank-biserial = {fn(rrb)}. "
              + _verdict(p, f"يوجد تغيّر دال إحصائيًا بين {n_x} و{n_y}", f"لا يوجد تغيّر دال إحصائيًا بين {n_x} و{n_y}", alpha))
    return {"ok": True, "family": "nonparam", "test": "Wilcoxon", "stat": float(res.statistic), "z": float(abs(z)) if np.isfinite(z) else np.nan,
            "p": p, "effect": {"r": float(r) if np.isfinite(r) else np.nan, "rank-biserial": rrb}, "table": tbl, "interp": interp}


# =============================================================== k مجموعات مستقلة
def _pair_name(a, b):
    return f"{a} ↔ {b}"


def welch_anova(arrs):
    k = len(arrs)
    n = np.array([len(a) for a in arrs], float)
    m = np.array([a.mean() for a in arrs])
    v = np.array([a.var(ddof=1) for a in arrs])
    if (v == 0).any():
        return np.nan, np.nan, np.nan, np.nan
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    A = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / W) ** 2 / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = A / B
    df2 = (k ** 2 - 1) / (3 * lam)
    return float(F), k - 1, float(df2), float(stats.f.sf(F, k - 1, df2))


def dunn_posthoc(groups, adjust="holm"):
    """اختبار Dunn البعدي (مع تصحيح الروابط). groups: dict اسم -> مصفوفة."""
    items = [(k, _clean(v)) for k, v in groups.items()]
    items = [(k, v) for k, v in items if len(v) >= 1]
    allv = np.concatenate([v for _, v in items])
    N = len(allv)
    ranks = stats.rankdata(allv)
    pos = 0
    mr = {}
    for k, v in items:
        mr[k] = ranks[pos:pos + len(v)].mean()
        pos += len(v)
    tt = _tie_term(allv)
    base = N * (N + 1) / 12.0 - tt / (12.0 * (N - 1))
    rows = []
    for (k1, v1), (k2, v2) in itertools.combinations(items, 2):
        se = np.sqrt(base * (1 / len(v1) + 1 / len(v2)))
        z = (mr[k1] - mr[k2]) / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append({"المقارنة": _pair_name(k1, k2), "فرق متوسط الرتب": mr[k1] - mr[k2], "Z": z, "p الخام": p,
                     "r = |Z|/√(n₁+n₂)": abs(z) / np.sqrt(len(v1) + len(v2)) if np.isfinite(z) else np.nan})
    t = pd.DataFrame(rows)
    if len(t):
        t["p المصحَّح"] = p_adjust(t["p الخام"].fillna(1.0).tolist(), adjust)
        t["التصحيح"] = ADJ_AR.get(adjust, adjust)
        t["القرار"] = np.where(t["p المصحَّح"] < ALPHA, "دال", "غير دال")
    return t


def _posthoc_text(ph, name, col="p المصحَّح", alpha=ALPHA):
    if ph is None or len(ph) == 0:
        return ""
    s = ph[ph[col] < alpha]
    if len(s) == 0:
        return f" المقارنات البعدية ({name}): لا فروق دالة بين أي زوجين."
    items = "؛ ".join(f"{r['المقارنة']} ({fp(r[col])})" for _, r in s.iterrows())
    return f" المقارنات البعدية ({name}): فروق دالة بين {items}؛ وبقية الأزواج غير دالة."


@_safe
def k_independent(groups, family="param", posthoc="auto", adjust="holm", alpha=ALPHA):
    """مقارنة k≥2 مجموعات مستقلة.
    param: ANOVA (و Welch ANOVA) + posthoc ∈ {auto, tukey, games-howell, holm, bonferroni}
    nonparam: Kruskal–Wallis + Dunn (adjust ∈ holm|bonferroni)."""
    g = {k: _clean(v) for k, v in groups.items()}
    g = {k: v for k, v in g.items() if len(v) >= 2}
    if len(g) < 2:
        return _fail("يلزم مجموعتان على الأقل تضم كل منهما قيمتين فأكثر.")
    names = list(g)
    arrs = [g[k] for k in names]
    k = len(arrs)
    N = sum(len(a) for a in arrs)
    allv = np.concatenate(arrs)
    if np.ptp(allv) == 0:
        return _fail("جميع القيم متطابقة.")
    desc = "؛ ".join(f"{n_}: {_desc_str(a)}" if family == "param" else f"{n_}: وسيط {fn(np.median(a))} (ن = {len(a)})"
                     for n_, a in g.items())
    if family == "param":
        gm = allv.mean()
        ssb = sum(len(a) * (a.mean() - gm) ** 2 for a in arrs)
        ssw = sum(((a - a.mean()) ** 2).sum() for a in arrs)
        dfb, dfw = k - 1, N - k
        if dfw < 1:
            return _fail("عدد القيم لا يكفي (يلزم N > k).")
        msb, msw = ssb / dfb, ssw / dfw
        F = msb / msw if msw > 0 else np.inf
        p = float(stats.f.sf(F, dfb, dfw))
        sst = ssb + ssw
        eta = ssb / sst
        omega = max(0.0, (ssb - dfb * msw) / (sst + msw))
        lev = levene_test(g, alpha=alpha)
        equal = lev["equal"] if lev.get("ok") else True
        Fw, d1w, d2w, pw = welch_anova(arrs)
        tbl = pd.DataFrame([
            {"المصدر": "بين المجموعات", "SS": ssb, "df": dfb, "MS": msb, "F": F, "p": p, "η²": eta, "ω²": omega},
            {"المصدر": "داخل المجموعات", "SS": ssw, "df": dfw, "MS": msw, "F": np.nan, "p": np.nan, "η²": np.nan, "ω²": np.nan},
            {"المصدر": "الكلي", "SS": sst, "df": N - 1, "MS": np.nan, "F": np.nan, "p": np.nan, "η²": np.nan, "ω²": np.nan},
            {"المصدر": "Welch ANOVA (قوي لعدم تجانس التباين)", "SS": np.nan, "df": f"{d1w}, {fn(d2w, 1)}", "MS": np.nan,
             "F": Fw, "p": pw, "η²": np.nan, "ω²": np.nan}])
        use_welch = (not equal) and np.isfinite(pw)
        main_p = pw if use_welch else p
        test = "Welch ANOVA" if use_welch else "ANOVA أحادي الاتجاه"
        # post-hoc
        ph_method = posthoc
        if posthoc == "auto":
            ph_method = "games-howell" if not equal else "tukey"
        ph = _posthoc_param(g, names, ph_method, adjust, alpha)
        stat_txt = (f"Welch F({d1w}, {fn(d2w, 1)}) = {fn(Fw)}" if use_welch else f"F({dfb}, {dfw}) = {fn(F)}")
        interp = (f"{test} لمقارنة {k} مجموعات ({desc}). {stat_txt}, {fp(main_p)}؛ η² = {fn(eta, 3)} "
                  f"(أثر {mag(eta, 'eta')})، ω² = {fn(omega, 3)}. "
                  + (lev["interp"] + " " if lev.get("ok") else "")
                  + _verdict(main_p, "يوجد فرق دال إحصائيًا بين متوسطات المجموعات على الأقل بين مجموعتين",
                             "لا يوجد دليل على اختلاف متوسطات المجموعات", alpha)
                  + (_posthoc_text(ph, ph.attrs.get("name", "")) if _sig(main_p, alpha) else
                     " لم تُفسَّر المقارنات البعدية لأن الاختبار الشامل غير دال (الجدول للاطلاع)."))
        return {"ok": True, "family": "param", "test": test, "stat": float(Fw if use_welch else F),
                "df": (d1w, d2w) if use_welch else (dfb, dfw), "p": float(main_p),
                "effect": {"η²": float(eta), "ω²": float(omega)}, "table": tbl, "posthoc": ph, "levene": lev,
                "interp": interp}
    # Kruskal–Wallis
    H, p = stats.kruskal(*arrs)
    H, p = float(H), float(p)
    eps2 = H * (N + 1) / (N ** 2 - 1)
    ranks = stats.rankdata(allv)
    pos, mrs = 0, []
    for a in arrs:
        mrs.append(ranks[pos:pos + len(a)].mean())
        pos += len(a)
    tbl = pd.DataFrame([{"الاختبار": "Kruskal–Wallis H", "H": H, "df": k - 1, "p": p, "ε²": eps2,
                         "N": N}])
    rk = pd.DataFrame({"المجموعة": names, "n": [len(a) for a in arrs], "الوسيط": [float(np.median(a)) for a in arrs],
                       "متوسط الرتب": mrs})
    ph = dunn_posthoc(g, adjust if adjust in ("holm", "bonferroni", "bh") else "holm")
    interp = (f"اختبار Kruskal–Wallis لمقارنة {k} مجموعات ({desc}). H({k - 1}) = {fn(H)}, {fp(p)}؛ "
              f"ε² = {fn(eps2, 3)} (أثر {mag(eps2, 'eps')}). "
              + _verdict(p, "يوجد فرق دال إحصائيًا في توزيعات/رتب المجموعات على الأقل بين مجموعتين",
                         "لا يوجد دليل على اختلاف رتب المجموعات", alpha)
              + (_posthoc_text(ph, f"Dunn بتصحيح {ADJ_AR.get(adjust, adjust)}") if _sig(p, alpha) else
                 " لم تُفسَّر المقارنات البعدية لأن الاختبار الشامل غير دال (الجدول للاطلاع)."))
    return {"ok": True, "family": "nonparam", "test": "Kruskal–Wallis", "stat": H, "df": k - 1, "p": p,
            "effect": {"ε²": float(eps2)}, "table": tbl, "ranks": rk, "posthoc": ph, "interp": interp}


def _posthoc_param(g, names, method, adjust, alpha=ALPHA):
    arrs = [g[k] for k in names]
    k = len(arrs)
    rows = []
    N = sum(len(a) for a in arrs)
    msw = sum(((a - a.mean()) ** 2).sum() for a in arrs) / (N - k)
    if method == "tukey":
        res = stats.tukey_hsd(*arrs)
        ci = res.confidence_interval(1 - alpha)
        label = "Tukey HSD"
        for i, j in itertools.combinations(range(k), 2):
            gg, _, _ = hedges_g(arrs[i], arrs[j])
            rows.append({"المقارنة": _pair_name(names[i], names[j]), "فرق المتوسطات": float(res.statistic[i, j]),
                         "CI أدنى": float(ci.low[i, j]), "CI أعلى": float(ci.high[i, j]),
                         "الإحصاءة": float(res.statistic[i, j] / np.sqrt(msw / 2 * (1 / len(arrs[i]) + 1 / len(arrs[j])))),
                         "p الخام": np.nan, "p المصحَّح": float(res.pvalue[i, j]), "Hedges g": gg})
    elif method == "games-howell":
        label = "Games–Howell"
        for i, j in itertools.combinations(range(k), 2):
            a, b = arrs[i], arrs[j]
            v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se = np.sqrt(v1 + v2)
            df = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1)) if se > 0 else np.nan
            diff = a.mean() - b.mean()
            q = abs(diff) / se * np.sqrt(2) if se > 0 else np.nan
            p = float(stats.studentized_range.sf(q, k, df)) if np.isfinite(q) else np.nan
            qc = stats.studentized_range.ppf(1 - alpha, k, df) if np.isfinite(df) else np.nan
            gg, _, _ = hedges_g(a, b)
            rows.append({"المقارنة": _pair_name(names[i], names[j]), "فرق المتوسطات": diff,
                         "CI أدنى": diff - qc / np.sqrt(2) * se, "CI أعلى": diff + qc / np.sqrt(2) * se,
                         "الإحصاءة": diff / se if se > 0 else np.nan, "p الخام": np.nan, "p المصحَّح": p, "Hedges g": gg})
    else:  # holm / bonferroni : Welch t
        label = "Welch t بتصحيح " + ("Bonferroni" if method == "bonferroni" else "Holm")
        for i, j in itertools.combinations(range(k), 2):
            a, b = arrs[i], arrs[j]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            df = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
            se = np.sqrt(v1 + v2)
            tc = stats.t.ppf(1 - alpha / 2, df)
            diff = a.mean() - b.mean()
            gg, _, _ = hedges_g(a, b)
            rows.append({"المقارنة": _pair_name(names[i], names[j]), "فرق المتوسطات": diff,
                         "CI أدنى": diff - tc * se, "CI أعلى": diff + tc * se, "الإحصاءة": float(t),
                         "p الخام": float(p), "p المصحَّح": np.nan, "Hedges g": gg})
        adj = p_adjust([r["p الخام"] for r in rows], "bonferroni" if method == "bonferroni" else "holm")
        for r, a_ in zip(rows, adj):
            r["p المصحَّح"] = a_
    t = pd.DataFrame(rows)
    t["الطريقة"] = label
    t["القرار"] = np.where(t["p المصحَّح"] < alpha, "دال", "غير دال")
    t.attrs["name"] = label
    return t


# =============================================================== k قياسات مترابطة
def friedman_manual(Y):
    """Friedman مع تصحيح الروابط (يعمل أيضًا عند k=2). Y: n×k. يعيد (chi2, df, p, W)."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    R = np.apply_along_axis(stats.rankdata, 1, Y)
    Rj = R.sum(0)
    chi = 12.0 / (n * k * (k + 1)) * (Rj ** 2).sum() - 3.0 * n * (k + 1)
    tie = sum(_tie_term(row) for row in Y)
    corr = 1.0 - tie / (n * k * (k ** 2 - 1))
    if corr <= 0:
        return np.nan, k - 1, np.nan, np.nan
    chi /= corr
    return float(chi), k - 1, float(stats.chi2.sf(chi, k - 1)), float(chi / (n * (k - 1)))


def _complete(df, cols):
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").dropna()


def _pairwise_related(Y, cols, family, adjust, alpha=ALPHA):
    rows = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        r = two_related(Y[:, i], Y[:, j], family, names=(str(cols[i]), str(cols[j])), alpha=alpha)
        if not r.get("ok"):
            rows.append({"المقارنة": f"{cols[i]} → {cols[j]}", "متوسط التغير": float(np.mean(Y[:, j] - Y[:, i])),
                         "الإحصاءة": np.nan, "p الخام": np.nan, "حجم الأثر": np.nan, "نوع الأثر": "", "ملاحظة": r["reason"]})
            continue
        eff = r["effect"]
        e_name, e_val = next(iter(eff.items()))
        rows.append({"المقارنة": f"{cols[i]} → {cols[j]}", "متوسط التغير": float(np.mean(Y[:, j] - Y[:, i])),
                     "الإحصاءة": r["stat"], "p الخام": r["p"], "حجم الأثر": e_val,
                     "نوع الأثر": {"d_z": "d_z", "r": "r", "rank-biserial": "rank-biserial"}.get(e_name, e_name), "ملاحظة": ""})
    t = pd.DataFrame(rows)
    if len(t):
        t["p المصحَّح"] = p_adjust(t["p الخام"].fillna(1.0).tolist(), adjust)
        t["التصحيح"] = ADJ_AR.get(adjust, adjust)
        t["القرار"] = np.where(t["p المصحَّح"] < alpha, "دال", "غير دال")
    return t


@_safe
def rm_oneway(data, cols=None, family="param", adjust="holm", alpha=ALPHA):
    """قياسات متكررة لمجموعة واحدة (الحالات الكاملة).
    param: RM-ANOVA مع Greenhouse–Geisser + t مترابط Holm ؛ nonparam: Friedman + Kendall W + Wilcoxon Holm."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    cols = list(cols) if cols is not None else list(df.columns)
    cc = _complete(df, cols)
    n, k = cc.shape
    if k < 2:
        return _fail("يلزم قياسان على الأقل.")
    if n < 3:
        return _fail(f"يلزم 3 حالات كاملة على الأقل (المتوفر {n}).")
    Y = cc.to_numpy(float)
    if np.ptp(Y) == 0:
        return _fail("جميع القيم متطابقة.")
    mean_tbl = pd.DataFrame({"القياس": cols, "n": n, "المتوسط": Y.mean(0), "الانحراف المعياري": Y.std(0, ddof=1),
                             "الوسيط": np.median(Y, axis=0)})
    if family == "param":
        grand = Y.mean()
        ss_t = n * ((Y.mean(0) - grand) ** 2).sum()
        ss_s = k * ((Y.mean(1) - grand) ** 2).sum()
        ss_tot = ((Y - grand) ** 2).sum()
        ss_e = ss_tot - ss_t - ss_s
        df1, df2 = k - 1, (n - 1) * (k - 1)
        if ss_e <= 1e-12:
            return _fail("لا يوجد تباين خطأ (الفروق بين القياسات متطابقة عند جميع الأفراد).")
        F = (ss_t / df1) / (ss_e / df2)
        p = float(stats.f.sf(F, df1, df2))
        eps, lb = gg_epsilon(np.cov(Y, rowvar=False)) if k > 2 else (1.0, 1.0)
        hf = hf_epsilon(eps, n, k) if k > 2 else 1.0
        p_gg = float(stats.f.sf(F, eps * df1, eps * df2))
        pe2 = ss_t / (ss_t + ss_e)
        ges = ss_t / (ss_t + ss_s + ss_e)
        tbl = pd.DataFrame([
            {"المصدر": "الزمن/القياس", "SS": ss_t, "df": df1, "MS": ss_t / df1, "F": F, "p": p, "ε (GG)": eps,
             "df (GG)": f"{eps * df1:.2f}, {eps * df2:.2f}", "p (GG)": p_gg, "η² جزئي": pe2, "η² المعمَّم": ges},
            {"المصدر": "الخطأ", "SS": ss_e, "df": df2, "MS": ss_e / df2},
            {"المصدر": "بين الأفراد", "SS": ss_s, "df": n - 1, "MS": ss_s / (n - 1)}])
        ph = _pairwise_related(Y, cols, "param", adjust, alpha)
        pmain = p_gg
        interp = (f"تحليل التباين للقياسات المتكررة (RM-ANOVA، ن = {n} حالة كاملة، {k} قياسات). "
                  f"معامل Greenhouse–Geisser ε = {fn(eps, 3)} (Huynh–Feldt = {fn(hf, 3)}). "
                  f"F({fn(eps * df1, 2)}, {fn(eps * df2, 2)}) = {fn(F)}, {fp(p_gg)} (مصحَّح GG؛ غير المصحَّح: F({df1}, {df2}), {fp(p)})؛ "
                  f"η² الجزئي = {fn(pe2, 3)} (أثر {mag(pe2, 'eta')})، η² المعمَّم = {fn(ges, 3)}. "
                  + _verdict(pmain, "يوجد تغيّر دال إحصائيًا عبر القياسات", "لا يوجد دليل على تغيّر المتوسطات عبر القياسات", alpha)
                  + (_posthoc_text(ph, f"t مترابط بتصحيح {ADJ_AR.get(adjust, adjust)}") if _sig(pmain, alpha) else ""))
        return {"ok": True, "family": "param", "test": "RM-ANOVA (GG)", "stat": float(F), "df": (df1, df2), "p": pmain,
                "p_uncorrected": p, "eps_gg": eps, "eps_hf": hf, "effect": {"η² جزئي": float(pe2), "η² معمَّم": float(ges)},
                "ss": {"time": ss_t, "subj": ss_s, "err": ss_e, "total": ss_tot}, "n": n,
                "table": tbl, "posthoc": ph, "means": mean_tbl, "interp": interp}
    chi, dfc, p, W = friedman_manual(Y)
    if not np.isfinite(chi):
        return _fail("جميع الرتب متطابقة؛ لا يمكن إجراء Friedman.")
    ranks = np.apply_along_axis(stats.rankdata, 1, Y).mean(0)
    tbl = pd.DataFrame([{"الاختبار": "Friedman χ²", "χ²": chi, "df": dfc, "p": p, "Kendall W": W, "n": n}])
    mean_tbl["متوسط الرتب"] = ranks
    ph = _pairwise_related(Y, cols, "nonparam", adjust, alpha)
    interp = (f"اختبار Friedman للقياسات المتكررة (ن = {n} حالة كاملة، {k} قياسات): χ²({dfc}) = {fn(chi)}, {fp(p)}؛ "
              f"معامل التوافق Kendall W = {fn(W, 3)} (أثر {mag(W, 'w')}). "
              + _verdict(p, "يوجد اختلاف دال إحصائيًا في رتب القياسات", "لا يوجد دليل على اختلاف رتب القياسات", alpha)
              + (_posthoc_text(ph, f"Wilcoxon بتصحيح {ADJ_AR.get(adjust, adjust)}") if _sig(p, alpha) else ""))
    return {"ok": True, "family": "nonparam", "test": "Friedman", "stat": chi, "df": dfc, "p": p,
            "effect": {"Kendall W": W}, "n": n, "table": tbl, "posthoc": ph, "means": mean_tbl, "interp": interp}


# =============================================================== التصميم المختلط
@_safe
def mixed_anova(wide, group_col, time_cols, alpha=ALPHA):
    """ANOVA مختلط: بين = المجموعة، داخل = الزمن (الحالات الكاملة) بصيغ يدوية؛ تصحيح GG للتأثيرات الداخلية."""
    cols = list(time_cols)
    d = wide[[group_col] + cols].copy()
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna()
    cnt = d[group_col].value_counts()
    keep = cnt[cnt >= 2].index
    d = d[d[group_col].isin(keep)]
    groups = [g for g in pd.unique(d[group_col])]
    a, k, N = len(groups), len(cols), len(d)
    if a < 2:
        return _fail("يلزم مجموعتان على الأقل تضم كل منهما لاعبَين (حالتين) كاملتين.")
    if k < 2:
        return _fail("يلزم قياسان زمنيان على الأقل.")
    if N - a < 1:
        return _fail("عدد الحالات الكاملة غير كافٍ.")
    Y = d[cols].to_numpy(float)
    gv = d[group_col].to_numpy()
    if np.ptp(Y) == 0:
        return _fail("جميع القيم متطابقة.")
    mi = Y.mean(1)
    grand = Y.mean()
    ss_g = ss_s = ss_cells = 0.0
    pooled = np.zeros((k, k))
    cell_means = {}
    for g in groups:
        idx = gv == g
        ng = idx.sum()
        mg = Y[idx].mean()
        ss_g += k * ng * (mg - grand) ** 2
        ss_s += k * ((mi[idx] - mg) ** 2).sum()
        cm = Y[idx].mean(0)
        cell_means[g] = (ng, cm)
        ss_cells += ng * ((cm - mg) ** 2).sum()
        Yc = Y[idx] - cm
        pooled += Yc.T @ Yc
    ss_t = N * ((Y.mean(0) - grand) ** 2).sum()
    ss_w = ((Y - mi[:, None]) ** 2).sum()
    ss_gt = ss_cells - ss_t
    ss_e = ss_w - ss_cells
    ss_tot = ((Y - grand) ** 2).sum()
    df_g, df_s = a - 1, N - a
    df_t, df_gt, df_e = k - 1, (a - 1) * (k - 1), (N - a) * (k - 1)
    pooled /= (N - a)
    eps, lb = gg_epsilon(pooled) if k > 2 else (1.0, 1.0)
    hf = hf_epsilon(eps, N, k, a) if k > 2 else 1.0
    if ss_s <= 1e-12 or ss_e <= 1e-12:
        return _fail("تباين الخطأ صفري؛ لا يمكن حساب F.")
    ms_s, ms_e = ss_s / df_s, ss_e / df_e
    F_g = (ss_g / df_g) / ms_s
    F_t = (ss_t / df_t) / ms_e
    F_gt = (ss_gt / df_gt) / ms_e
    p_g = float(stats.f.sf(F_g, df_g, df_s))
    p_t = float(stats.f.sf(F_t, df_t, df_e))
    p_gt = float(stats.f.sf(F_gt, df_gt, df_e))
    p_t_gg = float(stats.f.sf(F_t, eps * df_t, eps * df_e))
    p_gt_gg = float(stats.f.sf(F_gt, eps * df_gt, eps * df_e))
    pe = lambda ss, err: ss / (ss + err)  # noqa: E731
    rows = [
        {"المصدر": "المجموعة (بين الأفراد)", "SS": ss_g, "df": df_g, "MS": ss_g / df_g, "F": F_g, "p": p_g, "ε": np.nan,
         "p (GG)": p_g, "η² جزئي": pe(ss_g, ss_s)},
        {"المصدر": "الخطأ (بين الأفراد)", "SS": ss_s, "df": df_s, "MS": ms_s},
        {"المصدر": "الزمن", "SS": ss_t, "df": df_t, "MS": ss_t / df_t, "F": F_t, "p": p_t, "ε": eps, "p (GG)": p_t_gg,
         "η² جزئي": pe(ss_t, ss_e)},
        {"المصدر": "المجموعة × الزمن", "SS": ss_gt, "df": df_gt, "MS": ss_gt / df_gt, "F": F_gt, "p": p_gt, "ε": eps,
         "p (GG)": p_gt_gg, "η² جزئي": pe(ss_gt, ss_e)},
        {"المصدر": "الخطأ (داخل الأفراد)", "SS": ss_e, "df": df_e, "MS": ms_e},
    ]
    tbl = pd.DataFrame(rows)
    cm_rows = []
    for g in groups:
        ng, cm = cell_means[g]
        sds = Y[gv == g].std(0, ddof=1)
        for c, m_, s_ in zip(cols, cm, sds):
            cm_rows.append({"المجموعة": g, "القياس": c, "n": ng, "المتوسط": m_, "الانحراف المعياري": s_})
    cells = pd.DataFrame(cm_rows)
    lev_rows = []
    for c in cols:
        lv = levene_test({g: d.loc[gv == g, c].to_numpy() for g in groups}, alpha=alpha)
        if lv.get("ok"):
            lev_rows.append({"القياس": c, "F": lv["W"], "df1": lv["df1"], "df2": lv["df2"], "p": lv["p"],
                             "التجانس": "متحقق" if lv["equal"] else "غير متحقق"})
    lev_tbl = pd.DataFrame(lev_rows)

    def line(name, F, d1, d2, p, ss, err, extra=""):
        return (f"{name}: F({d1}, {d2}) = {fn(F)}, {fp(p)}, η² جزئي = {fn(pe(ss, err), 3)} "
                f"(أثر {mag(pe(ss, err), 'eta')}){extra}.")
    gg_note = f" [مصحَّح GG بـ ε = {fn(eps, 3)}]" if k > 2 else ""
    parts = [
        f"ANOVA مختلط (بين: المجموعة {a} مستويات؛ داخل: الزمن {k} قياسات؛ ن = {N} حالة كاملة"
        + ("، التصميم غير متوازن ومجاميع المربعات موزونة" if len(set(cnt[keep])) > 1 else "") + ").",
        line("التفاعل مجموعة × زمن (الأهم لفعالية البرنامج)", F_gt, fn(eps * df_gt, 2), fn(eps * df_e, 2), p_gt_gg, ss_gt, ss_e, gg_note),
        line("الأثر الرئيس للزمن", F_t, fn(eps * df_t, 2), fn(eps * df_e, 2), p_t_gg, ss_t, ss_e, gg_note),
        line("الأثر الرئيس للمجموعة", F_g, df_g, df_s, p_g, ss_g, ss_s),
    ]
    if _sig(p_gt_gg, alpha):
        parts.append("التفاعل دال: نمط التغير عبر الزمن يختلف بين المجموعات، فيُفسَّر الأثران الرئيسان بحذر وتُفحص التأثيرات البسيطة (مقارنة المجموعات في كل زمن، والتغير داخل كل مجموعة) مع تصحيح المقارنات المتعددة.")
    else:
        parts.append("التفاعل غير دال: لا يوجد دليل كافٍ على اختلاف نمط التغير عبر الزمن بين المجموعات (وهذا لا يثبت تماثله، فقد تكون القدرة الإحصائية محدودة).")
    parts.append(f"Greenhouse–Geisser ε = {fn(eps, 3)}، Huynh–Feldt = {fn(hf, 3)}. "
                 "لم يُحسب Box's M/Mauchly؛ يُعتمد تصحيح GG احتياطًا. تُعرض نتائج ليفين لكل قياس أدناه.")
    return {"ok": True, "family": "param", "test": "ANOVA مختلط", "table": tbl, "cells": cells, "levene": lev_tbl,
            "eps_gg": eps, "eps_hf": hf, "N": N, "groups": groups, "time_cols": cols,
            "ss": {"G": ss_g, "S/G": ss_s, "T": ss_t, "GT": ss_gt, "E": ss_e, "total": ss_tot},
            "F": {"G": F_g, "T": F_t, "GT": F_gt}, "p": {"G": p_g, "T": p_t_gg, "GT": p_gt_gg},
            "p_unc": {"G": p_g, "T": p_t, "GT": p_gt},
            "interp": " ".join(parts)}


@_safe
def mixed_simple_effects(wide, group_col, time_cols, family="param", adjust="holm", alpha=ALPHA):
    """التأثيرات البسيطة: (1) مقارنة المجموعات عند كل زمن، (2) التغير الزمني داخل كل مجموعة."""
    cols = list(time_cols)
    rows_b, rows_w = [], []
    for c in cols:
        grp = {g: pd.to_numeric(s[c], errors="coerce").dropna().to_numpy() for g, s in wide.groupby(group_col, sort=False)}
        r = k_independent(grp, family, adjust=adjust, alpha=alpha)
        if r.get("ok"):
            dfs = r["df"]
            rows_b.append({"القياس": c, "الاختبار": r["test"], "الإحصاءة": r["stat"],
                           "df": ", ".join(fn(x, 1 if isinstance(x, float) and x != int(x) else 0) for x in dfs) if isinstance(dfs, tuple) else dfs,
                           "p الخام": r["p"], "حجم الأثر": next(iter(r["effect"].values())), "نوع الأثر": next(iter(r["effect"].keys()))})
    for g, s in wide.groupby(group_col, sort=False):
        r = rm_oneway(s, cols, family, adjust, alpha)
        if r.get("ok"):
            rows_w.append({"المجموعة": g, "الاختبار": r["test"], "n": r["n"], "الإحصاءة": r["stat"],
                           "p الخام": r["p"], "حجم الأثر": next(iter(r["effect"].values())), "نوع الأثر": next(iter(r["effect"].keys())),
                           "posthoc": r["posthoc"]})
    tb = pd.DataFrame(rows_b)
    if len(tb):
        tb["p المصحَّح"] = p_adjust(tb["p الخام"].tolist(), adjust)
        tb["القرار"] = np.where(tb["p المصحَّح"] < alpha, "دال", "غير دال")
    tw = pd.DataFrame(rows_w)
    posts = {}
    if len(tw):
        posts = {r["المجموعة"]: r["posthoc"] for _, r in tw.iterrows()}
        tw = tw.drop(columns=["posthoc"])
        tw["p المصحَّح"] = p_adjust(tw["p الخام"].tolist(), adjust)
        tw["القرار"] = np.where(tw["p المصحَّح"] < alpha, "دال", "غير دال")
    sb = [r["القياس"] for _, r in tb.iterrows() if r["القرار"] == "دال"] if len(tb) else []
    sw = [r["المجموعة"] for _, r in tw.iterrows() if r["القرار"] == "دال"] if len(tw) else []
    interp = (f"التأثيرات البسيطة بتصحيح {ADJ_AR.get(adjust, adjust)} داخل كل عائلة. "
              + ("فروق دالة بين المجموعات عند: " + "، ".join(sb) + ". " if sb else "لا فروق دالة بين المجموعات في أي زمن. ")
              + ("تغيّر زمني دال داخل: " + "، ".join(map(str, sw)) + "." if sw else "لا تغيّر زمني دال داخل أي مجموعة."))
    return {"ok": True, "between_at_time": tb, "within_group": tw, "within_posthoc": posts, "interp": interp}


@_safe
def mixed_nonparametric(wide, group_col, time_cols, adjust="holm", alpha=ALPHA):
    """بديل لامعلمي للتصميم المختلط: Kruskal–Wallis على درجات التغير (Tk − T0) لكل زمن + Friedman داخل كل مجموعة."""
    cols = list(time_cols)
    if len(cols) < 2:
        return _fail("يلزم قياسان على الأقل.")
    base = cols[0]
    kw_rows, dunn = [], {}
    for c in cols[1:]:
        ch = pd.to_numeric(wide[c], errors="coerce") - pd.to_numeric(wide[base], errors="coerce")
        grp = {g: ch[wide[group_col] == g].dropna().to_numpy() for g in pd.unique(wide[group_col])}
        r = k_independent(grp, "nonparam", adjust=adjust, alpha=alpha)
        if r.get("ok"):
            kw_rows.append({"درجة التغير": f"{c} − {base}", "H": r["stat"], "df": r["df"], "p الخام": r["p"], "ε²": r["effect"]["ε²"]})
            dunn[c] = r["posthoc"]
    kw = pd.DataFrame(kw_rows)
    if len(kw):
        kw["p المصحَّح"] = p_adjust(kw["p الخام"].tolist(), adjust)
        kw["القرار"] = np.where(kw["p المصحَّح"] < alpha, "دال", "غير دال")
    fr_rows, fr_post = [], {}
    for g, s in wide.groupby(group_col, sort=False):
        r = rm_oneway(s, cols, "nonparam", adjust, alpha)
        if r.get("ok"):
            fr_rows.append({"المجموعة": g, "n": r["n"], "χ²": r["stat"], "df": r["df"], "p الخام": r["p"], "Kendall W": r["effect"]["Kendall W"]})
            fr_post[g] = r["posthoc"]
    fr = pd.DataFrame(fr_rows)
    if len(fr):
        fr["p المصحَّح"] = p_adjust(fr["p الخام"].tolist(), adjust)
        fr["القرار"] = np.where(fr["p المصحَّح"] < alpha, "دال", "غير دال")
    if not len(kw) and not len(fr):
        return _fail("بيانات غير كافية للتحليل اللامعلمي.")
    sk = [r["درجة التغير"] for _, r in kw.iterrows() if r["القرار"] == "دال"] if len(kw) else []
    sf = [str(r["المجموعة"]) for _, r in fr.iterrows() if r["القرار"] == "دال"] if len(fr) else []
    interp = ("تنبيه: لا يوجد بديل لامعلمي دقيق لـ ANOVA المختلط؛ لذا قُدِّم تحليلان متكاملان: "
              "(1) اختبار Kruskal–Wallis على درجات التغير (Tk − T0) لمقارنة المجموعات في مقدار التغير (وهو يقابل التفاعل مجموعة × زمن)، "
              "(2) اختبار Friedman داخل كل مجموعة لتقييم التغير الزمني. "
              + (f"اختلفت المجموعات دالًّا إحصائيًا في درجات التغير: {'، '.join(sk)}. " if sk else "لم تختلف المجموعات دالًّا إحصائيًا في درجات التغير بعد التصحيح. ")
              + (f"وكان التغير الزمني دالًّا داخل: {'، '.join(sf)}." if sf else "ولم يكن التغير الزمني دالًّا داخل أي مجموعة بعد التصحيح.")
              + f" التصحيح المستخدم: {ADJ_AR.get(adjust, adjust)}.")
    return {"ok": True, "family": "nonparam", "test": "Kruskal على التغير + Friedman", "kw_change": kw, "dunn": dunn,
            "friedman": fr, "friedman_post": fr_post, "interp": interp}


# =============================================================== ANCOVA
def _ols(X, y):
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    res = y - X @ beta
    return beta, float(res @ res), XtXi, int(np.linalg.matrix_rank(X)), res


@_safe
def ancova(wide, group_col, base_col, post_col, rank=False, adjust="holm", alpha=ALPHA):
    """ANCOVA: post ~ group + baseline. rank=True → ANCOVA على الرتب (بديل لامعلمي)."""
    d = wide[[group_col, base_col, post_col]].copy()
    d[[base_col, post_col]] = d[[base_col, post_col]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna()
    cnt = d[group_col].value_counts()
    d = d[d[group_col].isin(cnt[cnt >= 2].index)]
    groups = list(pd.unique(d[group_col]))
    a, n = len(groups), len(d)
    if a < 2 or n < a + 3:
        return _fail("بيانات غير كافية لـ ANCOVA (يلزم مجموعتان وعدد حالات أكبر من عدد المجموعات + 2).")
    y = d[post_col].to_numpy(float)
    x = d[base_col].to_numpy(float)
    raw_y = y.copy()
    if rank:
        y, x = stats.rankdata(y), stats.rankdata(x)
    if np.ptp(y) == 0 or np.ptp(x) == 0:
        return _fail("القياس القبلي أو البعدي ثابت القيمة.")
    gv = d[group_col].to_numpy()
    D = np.column_stack([(gv == g).astype(float) for g in groups[1:]])
    one = np.ones(n)
    Xf = np.column_stack([one, D, x])
    Xg = np.column_stack([one, x])        # بدون المجموعة
    Xc = np.column_stack([one, D])        # بدون المتغير المصاحب
    Xi = np.column_stack([Xf, D * x[:, None]])
    beta, rss_f, XtXi, rk_f, res = _ols(Xf, y)
    _, rss_g, _, _, _ = _ols(Xg, y)
    _, rss_c, _, _, _ = _ols(Xc, y)
    _, rss_i, _, rk_i, _ = _ols(Xi, y)
    df_e = n - rk_f
    if df_e < 1 or rss_f <= 1e-12:
        return _fail("درجات الحرية للخطأ غير كافية.")
    mse = rss_f / df_e
    df_grp = a - 1
    F_g = ((rss_g - rss_f) / df_grp) / mse
    p_g = float(stats.f.sf(F_g, df_grp, df_e))
    F_c = (rss_c - rss_f) / mse
    p_c = float(stats.f.sf(F_c, 1, df_e))
    pe_g = (rss_g - rss_f) / (rss_g - rss_f + rss_f)
    pe_c = (rss_c - rss_f) / rss_c
    # تجانس الميول
    df_i = rk_i - rk_f
    if df_i > 0 and n - rk_i > 0 and rss_i > 1e-12:
        F_s = ((rss_f - rss_i) / df_i) / (rss_i / (n - rk_i))
        p_s = float(stats.f.sf(F_s, df_i, n - rk_i))
        slope = {"F": float(F_s), "df": (df_i, n - rk_i), "p": p_s, "homogeneous": bool(p_s > alpha)}
    else:
        slope = {"F": np.nan, "df": (df_i, n - rk_i), "p": np.nan, "homogeneous": None}
    xm = x.mean()
    Ls = {}
    for i, g in enumerate(groups):
        L = np.zeros(Xf.shape[1])
        L[0] = 1
        if i > 0:
            L[i] = 1
        L[-1] = xm
        Ls[g] = L
    tc = stats.t.ppf(1 - alpha / 2, df_e)
    adj_rows = []
    for g in groups:
        L = Ls[g]
        m_ = float(L @ beta)
        se = float(np.sqrt(mse * L @ XtXi @ L))
        idx = gv == g
        adj_rows.append({"المجموعة": g, "n": int(idx.sum()),
                         ("متوسط الرتب الخام" if rank else "المتوسط الخام (بعدي)"): float((y if rank else raw_y)[idx].mean()),
                         ("متوسط الرتب المعدَّل" if rank else "المتوسط المعدَّل"): m_, "الخطأ المعياري": se,
                         "CI أدنى": m_ - tc * se, "CI أعلى": m_ + tc * se})
    adj_tbl = pd.DataFrame(adj_rows)
    pw = []
    for g1, g2 in itertools.combinations(groups, 2):
        L = Ls[g1] - Ls[g2]
        diff = float(L @ beta)
        se = float(np.sqrt(mse * L @ XtXi @ L))
        t = diff / se if se > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), df_e)) if np.isfinite(t) else np.nan
        pw.append({"المقارنة": _pair_name(g1, g2), "فرق المتوسطات المعدَّلة": diff, "SE": se, "t": t, "df": df_e,
                   "CI أدنى": diff - tc * se, "CI أعلى": diff + tc * se, "p الخام": p})
    pw = pd.DataFrame(pw)
    pw["p المصحَّح"] = p_adjust(pw["p الخام"].fillna(1).tolist(), adjust)
    pw["التصحيح"] = ADJ_AR.get(adjust, adjust)
    pw["القرار"] = np.where(pw["p المصحَّح"] < alpha, "دال", "غير دال")
    tbl = pd.DataFrame([
        {"المصدر": "القياس القبلي (المتغير المصاحب)", "SS": rss_c - rss_f, "df": 1, "MS": rss_c - rss_f, "F": F_c, "p": p_c, "η² جزئي": pe_c},
        {"المصدر": "المجموعة (بعد ضبط القبلي)", "SS": rss_g - rss_f, "df": df_grp, "MS": (rss_g - rss_f) / df_grp, "F": F_g, "p": p_g, "η² جزئي": pe_g},
        {"المصدر": "الخطأ", "SS": rss_f, "df": df_e, "MS": mse}])
    shap = stats.shapiro(res)[1] if n >= 3 and np.ptp(res) > 0 else np.nan
    lev = levene_test({g: res[gv == g] for g in groups}, alpha=alpha)
    title = "ANCOVA على الرتب (بديل لامعلمي)" if rank else "ANCOVA"
    slope_txt = ("" if not np.isfinite(slope["p"]) else
                 f" افتراض تجانس الميول (تفاعل المجموعة × القبلي): F({slope['df'][0]}, {slope['df'][1]}) = {fn(slope['F'])}, {fp(slope['p'])} — "
                 + ("متحقق." if slope["homogeneous"] else "غير متحقق؛ فتعديل القياس القبلي بميل مشترك قد يكون مضللًا ويُفضَّل نموذج يتضمن التفاعل أو تحليل التغير."))
    interp = (f"{title}: المتغير التابع {post_col} والمصاحب {base_col} (ن = {n}، {a} مجموعات). "
              f"أثر المجموعة بعد ضبط القياس القبلي: F({df_grp}, {df_e}) = {fn(F_g)}, {fp(p_g)}, η² جزئي = {fn(pe_g, 3)} (أثر {mag(pe_g, 'eta')}). "
              f"أثر المصاحب: F(1, {df_e}) = {fn(F_c)}, {fp(p_c)}." + slope_txt + " "
              + _verdict(p_g, "تختلف المتوسطات المعدَّلة دالًّا إحصائيًا بين المجموعات", "لا يوجد دليل على اختلاف المتوسطات المعدَّلة بين المجموعات", alpha)
              + (_posthoc_text(pw, f"مقارنات المتوسطات المعدَّلة بتصحيح {ADJ_AR.get(adjust, adjust)}") if _sig(p_g, alpha) else "")
              + (" ملاحظة: في ANCOVA على الرتب تُقارن الرتب لا القيم الأصلية." if rank else ""))
    return {"ok": True, "family": "nonparam" if rank else "param", "test": title, "table": tbl, "adjusted": adj_tbl,
            "pairwise": pw, "slopes": slope, "F": float(F_g), "df": (df_grp, df_e), "p": p_g,
            "partial_eta2": float(pe_g), "resid_shapiro_p": float(shap) if np.isfinite(shap) else np.nan,
            "resid_levene": lev, "n": n, "interp": interp}


# =============================================================== الارتباط والانحدار
def _corr_one(x, y, method):
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.kendalltau(x, y)
    return float(r), float(p)


def _corr_ci(r, n, method, alpha=ALPHA):
    if method == "kendall" or n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3) if method == "pearson" else np.sqrt((1 + r ** 2 / 2) / (n - 3))  # Bonett–Wright
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


METHOD_AR = {"pearson": "بيرسون", "spearman": "سبيرمان", "kendall": "كيندال تاو-b"}


@_safe
def correlation_pair(x, y, method="pearson", alpha=ALPHA, names=("X", "Y")):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4:
        return _fail("يلزم 4 أزواج كاملة على الأقل.")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return _fail("أحد المتغيرين ثابت القيمة.")
    r, p = _corr_one(x, y, method)
    lo, hi = _corr_ci(r, n, method, alpha)
    sym = {"pearson": "r", "spearman": "ρ", "kendall": "τ-b"}[method]
    ci = f" (95% CI [{fn(lo)}, {fn(hi)}])" if np.isfinite(lo) else ""
    strength = mag(r, "r")
    interp = (f"ارتباط {METHOD_AR[method]} بين {names[0]} و{names[1]} (ن = {n}): {sym} = {fn(r)}{ci}, {fp(p)}؛ "
              f"علاقة {'طردية' if r > 0 else 'عكسية'} {strength}ة (حسب معايير كوهين). "
              + _verdict(p, "الارتباط دال إحصائيًا", "الارتباط غير دال إحصائيًا", alpha)
              + " الارتباط لا يعني السببية.")
    return {"ok": True, "r": r, "p": p, "n": n, "ci": (lo, hi), "interp": interp}


@_safe
def correlation_matrix(df, cols, method="pearson", adjust="none", alpha=ALPHA):
    """مصفوفة ارتباط مع p (حذف زوجي) وجدول أزواج بحدود الثقة (بيرسون: Fisher z؛ سبيرمان: Bonett–Wright)."""
    cols = list(cols)
    if len(cols) < 2:
        return _fail("اختر متغيرين على الأقل.")
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    R = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    P = pd.DataFrame(np.nan, index=cols, columns=cols)
    Nn = pd.DataFrame(0, index=cols, columns=cols)
    rows = []
    for a, b in itertools.combinations(cols, 2):
        s = data[[a, b]].dropna()
        n = len(s)
        Nn.loc[a, b] = Nn.loc[b, a] = n
        if n < 4 or s[a].nunique() < 2 or s[b].nunique() < 2:
            R.loc[a, b] = R.loc[b, a] = np.nan
            continue
        r, p = _corr_one(s[a].to_numpy(), s[b].to_numpy(), method)
        lo, hi = _corr_ci(r, n, method, alpha)
        R.loc[a, b] = R.loc[b, a] = r
        P.loc[a, b] = P.loc[b, a] = p
        rows.append({"المتغير 1": a, "المتغير 2": b, "n": n, "المعامل": r, "CI أدنى": lo, "CI أعلى": hi, "p الخام": p})
    pairs = pd.DataFrame(rows)
    if not len(pairs):
        return _fail("لا توجد أزواج كافية (يلزم 4 أزواج كاملة على الأقل).")
    pairs["p المصحَّح"] = p_adjust(pairs["p الخام"].tolist(), adjust)
    pairs["القرار"] = np.where(pairs["p المصحَّح"] < alpha, "دال", "غير دال")
    pairs["القوة"] = pairs["المعامل"].map(lambda v: mag(v, "r"))
    if adjust != "none":
        for _, r_ in pairs.iterrows():
            P.loc[r_["المتغير 1"], r_["المتغير 2"]] = P.loc[r_["المتغير 2"], r_["المتغير 1"]] = r_["p المصحَّح"]
    ns = int((pairs["القرار"] == "دال").sum())
    top = pairs.reindex(pairs["المعامل"].abs().sort_values(ascending=False).index).iloc[0]
    interp = (f"مصفوفة ارتباط {METHOD_AR[method]} (عدد الأزواج المختبرة = {len(pairs)}، حذف زوجي للقيم المفقودة، التصحيح: {ADJ_AR.get(adjust, adjust)}): "
              f"عدد الأزواج الدالة عند α = {alpha:g} هو {ns}. أقوى ارتباط بين «{top['المتغير 1']}» و«{top['المتغير 2']}» "
              f"({fn(top['المعامل'])}، {fp(top['p المصحَّح'])}). الارتباط لا يعني السببية، وتتضخم احتمالات الخطأ من النوع الأول مع كثرة الأزواج.")
    return {"ok": True, "r": R, "p": P, "n": Nn, "pairs": pairs, "interp": interp}


@_safe
def linear_regression(df, y_col, x_cols, alpha=ALPHA):
    """انحدار خطي بسيط/متعدد (OLS) بـ numpy: معاملات، SE، t، p، CI، β معيارية، VIF، R²، R² المعدل، F."""
    x_cols = list(x_cols)
    d = df[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n, p_ = len(d), len(x_cols)
    if p_ < 1:
        return _fail("اختر متغيرًا مستقلًا واحدًا على الأقل.")
    if n < p_ + 3:
        return _fail(f"عدد الحالات الكاملة ({n}) قليل لنموذج بـ {p_} متغيرات (يلزم {p_ + 3} على الأقل).")
    y = d[y_col].to_numpy(float)
    X = np.column_stack([np.ones(n), d[x_cols].to_numpy(float)])
    if np.ptp(y) == 0 or np.linalg.matrix_rank(X) < X.shape[1]:
        return _fail("المتغير التابع ثابت أو توجد مشكلة ازدواج خطي تام بين المتغيرات المستقلة.")
    beta, rss, XtXi, rk, res = _ols(X, y)
    dfe = n - X.shape[1]
    mse = rss / dfe
    se = np.sqrt(np.diag(XtXi) * mse)
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dfe)
    tc = stats.t.ppf(1 - alpha / 2, dfe)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / sst
    adj = 1 - (1 - r2) * (n - 1) / dfe
    F = ((sst - rss) / p_) / mse
    pF = float(stats.f.sf(F, p_, dfe))
    sx = d[x_cols].std(ddof=1).to_numpy()
    std_b = np.r_[np.nan, beta[1:] * sx / y.std(ddof=1)]
    vif = [np.nan]
    for j, c in enumerate(x_cols):
        if p_ == 1:
            vif.append(1.0)
            continue
        others = [i for i in range(p_) if i != j]
        Xo = np.column_stack([np.ones(n), d[x_cols].to_numpy(float)[:, others]])
        _, r_, _, _, _ = _ols(Xo, d[c].to_numpy(float))
        sst_j = ((d[c] - d[c].mean()) ** 2).sum()
        r2j = 1 - r_ / sst_j if sst_j > 0 else np.nan
        vif.append(1 / (1 - r2j) if r2j < 1 else np.inf)
    names = ["الثابت (Intercept)"] + x_cols
    tbl = pd.DataFrame({"المتغير": names, "B": beta, "SE": se, "β معياري": std_b, "t": t, "p": pv,
                        "CI أدنى": beta - tc * se, "CI أعلى": beta + tc * se, "VIF": vif})
    sh = stats.shapiro(res)[1] if n >= 3 and np.ptp(res) > 0 else np.nan
    model = pd.DataFrame([{"n": n, "R": np.sqrt(max(r2, 0)), "R²": r2, "R² المعدَّل": adj, "F": F, "df1": p_, "df2": dfe,
                           "p (F)": pF, "الخطأ المعياري للتقدير": np.sqrt(mse), "Shapiro للبواقي p": sh}])
    if p_ == 1:
        b = beta[1]
        interp = (f"انحدار خطي بسيط ({y_col} على {x_cols[0]}، ن = {n}): الميل B = {fn(b, 3)} (95% CI [{fn(b - tc * se[1], 3)}, {fn(b + tc * se[1], 3)}])، "
                  f"t({dfe}) = {fn(t[1])}, {fp(pv[1])}؛ الثابت = {fn(beta[0], 3)}؛ R² = {fn(r2, 3)} (يفسر {fn(r2 * 100, 1)}% من التباين). "
                  + _verdict(pv[1], "الميل يختلف دالًّا إحصائيًا عن الصفر", "لا يوجد دليل على ميل مختلف عن الصفر", alpha))
    else:
        sigv = [c for c, pp in zip(x_cols, pv[1:]) if pp < alpha]
        interp = (f"انحدار خطي متعدد ({y_col} على {p_} متغيرات، ن = {n}): F({p_}, {dfe}) = {fn(F)}, {fp(pF)}؛ "
                  f"R² = {fn(r2, 3)}، R² المعدَّل = {fn(adj, 3)}. "
                  + (f"المتغيرات ذات المعاملات الدالة: {'، '.join(sigv)}. " if sigv else "لا يوجد معامل دال عند α = 0.05. ")
                  + _verdict(pF, "النموذج دال إحصائيًا ككل", "النموذج ككل غير دال إحصائيًا", alpha)
                  + (" تنبيه: VIF > 5 يشير إلى ازدواج خطي." if np.nanmax(np.array(vif[1:], dtype=float)) > 5 else "")
                  + (" تنبيه: عدد الحالات لكل متغير قليل (<10) فالنموذج غير مستقر." if n / p_ < 10 else ""))
    return {"ok": True, "coef": tbl, "model": model, "r2": float(r2), "adj_r2": float(adj), "F": float(F), "p": pF,
            "n": n, "fitted": X @ beta, "resid": res, "interp": interp}


# =============================================================== الجداول التكرارية
@_safe
def chi_square_test(table, alpha=ALPHA, method="auto"):
    """table: DataFrame تكرارات (صفوف × أعمدة). method: auto | chi2 | fisher."""
    t = pd.DataFrame(table).astype(float)
    t = t.loc[t.sum(1) > 0, t.sum(0) > 0]
    r_, c_ = t.shape
    if r_ < 2 or c_ < 2:
        return _fail("يلزم صفان وعمودان على الأقل (فئتان فأكثر في كل متغير).")
    obs = t.to_numpy()
    n = obs.sum()
    chi2, p, dof, exp = stats.chi2_contingency(obs, correction=False)
    low = int((exp < 5).sum())
    pct_low = low / exp.size
    min_exp = float(exp.min())
    V = float(np.sqrt(chi2 / (n * (min(r_, c_) - 1)))) if n > 0 else np.nan
    out = {"ok": True, "n": int(n), "chi2": float(chi2), "df": int(dof), "p_chi2": float(p), "cramers_v": V,
           "min_expected": min_exp, "pct_expected_lt5": pct_low}
    exp_df = pd.DataFrame(exp, index=t.index, columns=t.columns)
    rowp = pd.DataFrame(obs / obs.sum(1, keepdims=True) * 100, index=t.index, columns=t.columns)
    warn = ""
    if low:
        warn = (f" تحذير: {low} خلية ({pct_low * 100:.0f}%) ذات تكرار متوقع أقل من 5 (أدنى قيمة {fn(min_exp)}) — "
                "شرط χ² غير متحقق جيدًا" + ("؛ يُفضَّل اختبار فيشر الدقيق." if (r_, c_) == (2, 2) else
                                            "؛ يُفضَّل دمج الفئات أو استخدام اختبار دقيق (Fisher–Freeman–Halton) في برنامج آخر."))
    use_fisher = (r_, c_) == (2, 2) and (method == "fisher" or (method == "auto" and low > 0))
    tables = {"observed": t, "expected": exp_df, "row_pct": rowp}
    out.update(tables)
    out["table"] = pd.DataFrame([{"الاختبار": "χ² بيرسون", "χ²": chi2, "df": dof, "p": p, "Cramér's V": V, "n": int(n),
                                  "أدنى تكرار متوقع": min_exp}])
    if (r_, c_) == (2, 2):
        chi_y = stats.chi2_contingency(obs, correction=True)
        odds, pf = stats.fisher_exact(obs)
        out["table"] = pd.concat([out["table"], pd.DataFrame([
            {"الاختبار": "χ² بتصحيح يتس للاستمرارية", "χ²": chi_y[0], "df": 1, "p": chi_y[1], "Cramér's V": V, "n": int(n), "أدنى تكرار متوقع": min_exp},
            {"الاختبار": "اختبار فيشر الدقيق", "χ²": np.nan, "df": np.nan, "p": pf, "Cramér's V": V, "n": int(n), "أدنى تكرار متوقع": min_exp}])],
            ignore_index=True)
        out["odds_ratio"] = float(odds)
        out["p_fisher"] = float(pf)
    if use_fisher:
        out.update(test="اختبار فيشر الدقيق", p=out["p_fisher"], stat=np.nan)
        txt = (f"اختبار فيشر الدقيق (جدول 2×2، ن = {int(n)}): {fp(out['p'])}؛ نسبة الأرجحية = {fn(out['odds_ratio'])}؛ "
               f"Cramér's V = {fn(V, 3)} (أثر {mag(V, 'r')}).")
    else:
        out.update(test="χ² بيرسون", p=float(p), stat=float(chi2))
        txt = (f"اختبار χ² للاستقلال (ن = {int(n)}): χ²({dof}) = {fn(chi2)}, {fp(p)}؛ Cramér's V = {fn(V, 3)} (أثر {mag(V, 'r')}).")
    out["interp"] = txt + warn + " " + _verdict(out["p"], "توجد علاقة (ارتباط) دالة إحصائيًا بين المتغيرين", "لا يوجد دليل على علاقة بين المتغيرين", alpha)
    return out


@_safe
def crosstab_test(df, row_col, col_col, alpha=ALPHA, method="auto"):
    d = df[[row_col, col_col]].dropna()
    if len(d) < 4:
        return _fail("عدد الحالات قليل جدًا للجدول التكراري.")
    ct = pd.crosstab(d[row_col], d[col_col])
    r = chi_square_test(ct, alpha, method)
    return r


@_safe
def responders_analysis(wide, group_col, base_col, post_col, min_drop=2.0, min_pct=0.30, direction="decrease", alpha=ALPHA):
    """المستجيبون: تحسّن ≥ min_drop (مطلق) أو ≥ min_pct (نسبي) ثم اختبار χ²/فيشر بين المجموعات."""
    d = wide[[group_col, base_col, post_col]].copy()
    d[[base_col, post_col]] = d[[base_col, post_col]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna()
    if len(d) < 4 or d[group_col].nunique() < 2:
        return _fail("بيانات غير كافية (يلزم مجموعتان و4 حالات كاملة على الأقل).")
    change = d[base_col] - d[post_col] if direction == "decrease" else d[post_col] - d[base_col]
    rel = change / d[base_col].replace(0, np.nan)
    d["مستجيب"] = np.where((change >= min_drop) | (rel >= min_pct), "مستجيب", "غير مستجيب")
    rows = []
    for g, s in d.groupby(group_col, sort=False):
        k_ = int((s["مستجيب"] == "مستجيب").sum())
        rows.append({"المجموعة": g, "n": len(s), "مستجيبون": k_, "غير مستجيبين": len(s) - k_, "النسبة %": 100.0 * k_ / len(s)})
    out = pd.DataFrame(rows)
    ct = pd.crosstab(d[group_col], d["مستجيب"])
    for c in ("مستجيب", "غير مستجيب"):
        if c not in ct.columns:
            ct[c] = 0
    r = chi_square_test(ct[["مستجيب", "غير مستجيب"]], alpha)
    res = {"ok": True, "table": out, "data": d}
    if r.get("ok"):
        res.update(test=r["test"], p=r["p"], chi=r, cramers_v=r["cramers_v"])
        res["interp"] = (f"عُدَّ اللاعب مستجيبًا إذا بلغ التحسن ≥ {min_drop:g} وحدة أو ≥ {min_pct * 100:.0f}% من القيمة القبلية. "
                         + "النسب: " + "، ".join(f"{r_['المجموعة']} {r_['النسبة %']:.0f}% ({r_['مستجيبون']}/{r_['n']})" for _, r_ in out.iterrows())
                         + ". " + r["interp"])
    else:
        res.update(p=np.nan, interp="تعذّر اختبار الفروق في نسب المستجيبين: " + r["reason"])
    return res


# =============================================================== القدرة وحجم العينة
def power_two_sample(d, n_per_group, alpha=ALPHA, ratio=1.0, paired_corr=None):
    """قدرة اختبار t لعينتين مستقلتين (ثنائي الطرف) عبر التوزيع t اللامركزي."""
    n1 = float(n_per_group)
    n2 = n1 * ratio
    if n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    nc = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def n_required_two_sample(d, power=0.80, alpha=ALPHA, ratio=1.0):
    if d == 0:
        return None
    n = 2
    while n < 100000:
        if power_two_sample(d, n, alpha, ratio) >= power:
            return n
        n += 1
    return None


def power_paired(dz, n, alpha=ALPHA):
    if n < 2:
        return float("nan")
    df = n - 1
    nc = abs(dz) * np.sqrt(n)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def n_required_paired(dz, power=0.80, alpha=ALPHA):
    if dz == 0:
        return None
    n = 3
    while n < 100000:
        if power_paired(dz, n, alpha) >= power:
            return n
        n += 1
    return None


def power_anova(f, k, n_per_group, alpha=ALPHA):
    """قدرة ANOVA أحادي الاتجاه (f كوهين) عبر F اللامركزي: λ = f²·N."""
    if k < 2 or n_per_group < 2:
        return float("nan")
    N = k * n_per_group
    df1, df2 = k - 1, N - k
    fc = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(fc, df1, df2, f ** 2 * N))


def n_required_anova(f, k, power=0.80, alpha=ALPHA):
    if f <= 0:
        return None
    n = 2
    while n < 100000:
        if power_anova(f, k, n, alpha) >= power:
            return n
        n += 1
    return None


def eta2_to_f(eta2):
    return float(np.sqrt(eta2 / (1 - eta2))) if 0 <= eta2 < 1 else float("nan")


def power_summary(kind, effect, n=None, k=3, power=0.80, alpha=ALPHA):
    """نص تفسيري عربي لحاسبة القدرة. kind: 'two' | 'paired' | 'anova'."""
    lab = {"two": "اختبار t لمجموعتين مستقلتين (d)", "paired": "اختبار t المترابط (d_z)", "anova": f"ANOVA أحادي الاتجاه ({k} مجموعات، f)"}[kind]
    if n:
        pw = {"two": lambda: power_two_sample(effect, n, alpha), "paired": lambda: power_paired(effect, n, alpha),
              "anova": lambda: power_anova(effect, k, n, alpha)}[kind]()
        return (f"{lab}: بحجم أثر {effect:g} و{n} لاعبًا {'في كل مجموعة' if kind != 'paired' else 'زوجًا'} وα = {alpha:g}، "
                f"القدرة الإحصائية = {pw * 100:.1f}%. " + ("القدرة ≥ 80% (مقبولة عرفًا)." if pw >= .8 else "القدرة أقل من 80% فاحتمال عدم كشف الأثر الحقيقي مرتفع."))
    req = {"two": lambda: n_required_two_sample(effect, power, alpha), "paired": lambda: n_required_paired(effect, power, alpha),
           "anova": lambda: n_required_anova(effect, k, power, alpha)}[kind]()
    if req is None:
        return "حجم الأثر المدخل غير صالح."
    return f"{lab}: لكشف أثر {effect:g} بقدرة {power * 100:.0f}% وα = {alpha:g} يلزم {req} {'في كل مجموعة' if kind != 'paired' else 'زوجًا'}."


# =============================================================== تصدير
def results_to_excel(tables, notes=None):
    """tables: dict اسم -> DataFrame ؛ notes: قائمة (عنوان، نص تفسير). يعيد bytes لملف xlsx."""
    bio = io.BytesIO()
    used = set()

    def sheet(name):
        s = re.sub(r"[\[\]\*\?/\\:]", "-", str(name))[:28] or "Sheet"
        base, i = s, 2
        while s.lower() in used:
            s = f"{base[:25]}_{i}"
            i += 1
        used.add(s.lower())
        return s
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        if notes:
            pd.DataFrame(notes, columns=["التحليل", "التفسير"]).to_excel(xw, sheet_name=sheet("التفسير"), index=False)
        wrote = bool(notes)
        for name, df in (tables or {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                df.to_excel(xw, sheet_name=sheet(name), index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)
                wrote = True
        if not wrote:
            pd.DataFrame({"ملاحظة": ["لا توجد نتائج بعد"]}).to_excel(xw, sheet_name="نتائج", index=False)
        for ws in xw.book.worksheets:
            ws.sheet_view.rightToLeft = True
            for col in ws.columns:
                w = max((len(str(c.value)) if c.value is not None else 0) for c in col[:50])
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, w + 2))
    return bio.getvalue()
