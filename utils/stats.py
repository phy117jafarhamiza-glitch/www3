# -*- coding: utf-8 -*-
"""
الوسائل الإحصائية المقترحة (3.9 في المقترح) — بدائل معلمية ولامعلمية لتصميم
ثلاث مجموعات × أربع نقاط قياس، مع حجم الأثر والتصحيح للمقارنات المتعددة.

ملاحظة: التحليل الرئيس الموصى به هو Mixed ANOVA أو النماذج الخطية المختلطة
(SPSS/Jamovi/R). هذه الصفحة تقدم تحليلاً أوليًا سريعًا مبنيًا على scipy:
وصفي، اعتدالية، تكافؤ قبلي، تغيّر داخل المجموعة، فروق التغير بين المجموعات
(ANOVA/Kruskal)، وANCOVA (ضبط القياس القبلي).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------- مساعدات
def holm(pvals):
    """تصحيح Holm–Bonferroni. يعيد قيم p المصحَّحة بنفس الترتيب."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wide_table(measurements: list, subjects: list, test_key: str, stages: list) -> pd.DataFrame:
    """جدول عريض: صف لكل لاعب وأعمدة لنقاط القياس."""
    sdf = pd.DataFrame(subjects)[["id", "group_name"]].rename(columns={"id": "subject_id"})
    if not measurements:
        return pd.DataFrame(columns=["subject_id", "group_name"] + stages)
    m = pd.DataFrame(measurements)
    m = m[m["test_key"] == test_key]
    piv = m.pivot_table(index="subject_id", columns="stage", values="value", aggfunc="first")
    for s in stages:
        if s not in piv.columns:
            piv[s] = np.nan
    piv = piv[stages].reset_index()
    return sdf.merge(piv, on="subject_id", how="left")


def descriptive(df: pd.DataFrame, stages: list) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby("group_name"):
        for s in stages:
            v = sub[s].dropna().to_numpy(float)
            if len(v) == 0:
                continue
            q1, q3 = np.percentile(v, [25, 75])
            rows.append({"المجموعة": g, "القياس": s, "n": len(v), "المتوسط": v.mean(),
                         "الانحراف": v.std(ddof=1) if len(v) > 1 else 0.0,
                         "الوسيط": float(np.median(v)), "Q1": q1, "Q3": q3})
    return pd.DataFrame(rows)


def normality(df: pd.DataFrame, stages: list) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby("group_name"):
        for s in stages:
            v = sub[s].dropna().to_numpy(float)
            if len(v) >= 3 and np.ptp(v) > 0:
                w, p = stats.shapiro(v)
                rows.append({"المجموعة": g, "القياس": s, "n": len(v), "W": w, "p": p, "طبيعي (p>.05)": bool(p > .05)})
    return pd.DataFrame(rows)


def _groups_values(df, col):
    return {g: sub[col].dropna().to_numpy(float) for g, sub in df.groupby("group_name") if sub[col].notna().sum() >= 1}


def between_groups(df: pd.DataFrame, col: str, force: str = "auto"):
    """مقارنة المجموعات في عمود واحد: ANOVA أو Kruskal–Wallis (تلقائيًا بحسب الاعتدالية)
    مع مقارنات بعدية Holm. يعيد قاموسًا بالنتائج."""
    vals = _groups_values(df, col)
    vals = {g: v for g, v in vals.items() if len(v) >= 2}
    if len(vals) < 2:
        return {"ok": False, "reason": "يلزم لاعبان على الأقل في مجموعتين"}
    normal = all(len(v) < 3 or np.ptp(v) == 0 or stats.shapiro(v)[1] > .05 for v in vals.values())
    use_param = (force == "param") or (force == "auto" and normal)
    arrs = list(vals.values())
    if use_param:
        stat, p = stats.f_oneway(*arrs)
        allv = np.concatenate(arrs)
        ss_b = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrs)
        ss_t = ((allv - allv.mean()) ** 2).sum()
        eff = {"η²": float(ss_b / ss_t) if ss_t > 0 else 0.0}
        test = "ANOVA أحادي الاتجاه"
    else:
        stat, p = stats.kruskal(*arrs)
        n = sum(len(a) for a in arrs)
        k = len(arrs)
        eff = {"ε²": float(max(0.0, (stat - k + 1) / (n - k))) if n > k else 0.0}
        test = "Kruskal–Wallis"
    pairs, raw_p = [], []
    for (g1, a), (g2, b) in itertools.combinations(vals.items(), 2):
        if use_param:
            t, pp = stats.ttest_ind(a, b, equal_var=False)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d = float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0
            pairs.append({"المقارنة": f"{g1} ↔ {g2}", "الاختبار": "Welch t", "الإحصاءة": float(t), "p": float(pp), "حجم الأثر (d)": d})
        else:
            u, pp = stats.mannwhitneyu(a, b, alternative="two-sided")
            r = 1 - 2 * u / (len(a) * len(b))
            pairs.append({"المقارنة": f"{g1} ↔ {g2}", "الاختبار": "Mann–Whitney", "الإحصاءة": float(u), "p": float(pp), "حجم الأثر (r)": float(r)})
        raw_p.append(pp)
    for rec, adj in zip(pairs, holm(raw_p)):
        rec["p (Holm)"] = adj
    return {"ok": True, "test": test, "stat": float(stat), "p": float(p), "effect": eff, "pairs": pairs, "normal": bool(normal)}


def within_group_change(df: pd.DataFrame, base: str, others: list):
    """تغيّر داخل كل مجموعة: القياس الأساسي مقابل كل قياس لاحق (t مترابط أو Wilcoxon)."""
    rows = []
    for g, sub in df.groupby("group_name"):
        for s in others:
            pair = sub[[base, s]].dropna()
            if len(pair) < 3:
                continue
            a, b = pair[base].to_numpy(float), pair[s].to_numpy(float)
            diff = b - a
            normal = np.ptp(diff) == 0 or stats.shapiro(diff)[1] > .05
            if normal and np.ptp(diff) > 0:
                t, p = stats.ttest_rel(a, b)
                dz = float(diff.mean() / diff.std(ddof=1))
                rows.append({"المجموعة": g, "المقارنة": f"{base} ← {s}", "n": len(pair), "الاختبار": "t مترابط",
                             "متوسط التغير": float(diff.mean()), "الإحصاءة": float(t), "p": float(p), "حجم الأثر (dz)": dz})
            else:
                try:
                    w, p = stats.wilcoxon(a, b)
                except ValueError:
                    continue
                n = len(pair)
                z = stats.norm.isf(p / 2)
                rows.append({"المجموعة": g, "المقارنة": f"{base} ← {s}", "n": n, "الاختبار": "Wilcoxon",
                             "متوسط التغير": float(diff.mean()), "الإحصاءة": float(w), "p": float(p),
                             "حجم الأثر (r)": float(z / np.sqrt(n))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p (Holm)"] = holm(out["p"].tolist())
    return out


def ancova(df: pd.DataFrame, base: str, post: str):
    """ANCOVA: القياس البعدي ~ المجموعة + القياس القبلي. يعيد F وp و η² الجزئي
    للمجموعة، والمتوسطات المعدَّلة عند متوسط القياس القبلي."""
    d = df[["group_name", base, post]].dropna()
    groups = sorted(d["group_name"].unique())
    if len(groups) < 2 or len(d) < len(groups) + 3:
        return {"ok": False, "reason": "بيانات غير كافية"}
    y = d[post].to_numpy(float)
    x = d[base].to_numpy(float)
    D = np.column_stack([(d["group_name"] == g).astype(float) for g in groups[1:]])
    Xf = np.column_stack([np.ones(len(d)), D, x])
    Xr = np.column_stack([np.ones(len(d)), x])

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), beta
    rss_f, beta = rss(Xf)
    rss_r, _ = rss(Xr)
    df1, df2 = len(groups) - 1, len(d) - Xf.shape[1]
    if df2 <= 0 or rss_f <= 0:
        return {"ok": False, "reason": "درجات حرية غير كافية"}
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p = float(stats.f.sf(F, df1, df2))
    eta = float((rss_r - rss_f) / rss_r) if rss_r > 0 else 0.0
    xm = x.mean()
    adj = {groups[0]: float(beta[0] + beta[-1] * xm)}
    for i, g in enumerate(groups[1:]):
        adj[g] = float(beta[0] + beta[1 + i] + beta[-1] * xm)
    return {"ok": True, "F": float(F), "df": (df1, df2), "p": p, "partial_eta2": eta, "adjusted_means": adj}


def responders(df: pd.DataFrame, base: str, post: str, min_drop=2.0, min_pct=0.30):
    """نسبة المستجيبين للألم: انخفاض ≥ نقطتين أو ≥ 30% (Farrar et al., 2001)."""
    rows = []
    d = df[["group_name", base, post]].dropna()
    for g, sub in d.groupby("group_name"):
        drop = sub[base] - sub[post]
        resp = (drop >= min_drop) | ((drop / sub[base].replace(0, np.nan)) >= min_pct)
        rows.append({"المجموعة": g, "n": len(sub), "مستجيبون": int(resp.sum()), "النسبة %": 100.0 * float(resp.mean())})
    out = pd.DataFrame(rows)
    p = None
    if len(out) >= 2 and out["n"].min() > 0:
        table = np.array([[r["مستجيبون"], r["n"] - r["مستجيبون"]] for r in rows])
        try:
            if len(out) == 2:
                p = float(stats.fisher_exact(table)[1])
            else:
                p = float(stats.chi2_contingency(table)[1])
        except ValueError:
            p = None
    return out, p


def baseline_equivalence(subjects: list, covariates: dict):
    """تكافؤ المجموعات في المتغيرات القبلية (العمر، BMI، ...)."""
    df = pd.DataFrame(subjects)
    rows = []
    for key, label in covariates.items():
        if key not in df.columns:
            continue
        r = between_groups(df.rename(columns={key: "_v"}), "_v")
        if r.get("ok"):
            rows.append({"المتغير": label, "الاختبار": r["test"], "p": r["p"], "متكافئ (p>.05)": bool(r["p"] > .05)})
    return pd.DataFrame(rows)
