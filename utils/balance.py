# -*- coding: utf-8 -*-
"""
مؤشرات التوازن العصبي العضلي والتحكم الحركي للرقبة ومؤشر NMCI المركّب (0–100).

تنبيه علمي: الحدود الافتراضية أدناه مبدئية (Placeholders) وقابلة للتعديل من الصفحة.
يُفضَّل تحديدها من مجموعة ضابطة سليمة أو من الأدبيات قبل اعتماد المؤشر في الأطروحة.
"""
import numpy as np

from utils.kinematics import symmetry_index

DEFAULT_CFG = {
    "ref_strength_ratio": 0.65,     # قوة المثنيات ÷ الباسطات (مرجع مبدئي)
    "ref_endurance_ratio": 0.40,    # تحمل المثنيات ÷ الباسطات (مرجع مبدئي)
    "sym_best": 5.0, "sym_worst": 30.0,          # % عدم التماثل (قوة)
    "ppt_best": 10.0, "ppt_worst": 40.0,         # % عدم تماثل PPT
    "ratio_best": 0.10, "ratio_worst": 0.50,     # انحراف نسبي عن المرجع
    "emg_best": 10.0, "emg_worst": 40.0,         # % MVC للعضلات السطحية أثناء CCFT
    "rmse_best": 2.0, "rmse_worst": 10.0,        # درجة
    "sparc_best": -1.6, "sparc_worst": -4.0,
    "coupled_best": 5.0, "coupled_worst": 35.0,  # %
    "velasym_best": 5.0, "velasym_worst": 30.0,  # %
}


def linear_score(x, best, worst):
    """100 عند best و0 عند worst (خطي، مقصوص)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = 100.0 * (x - worst) / (best - worst)
    return float(min(100.0, max(0.0, s)))


def _g(m, k):
    v = m.get(k)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def balance_metrics(m: dict, cfg=None):
    """m: قاموس مفاتيح القياس -> قيمة. يعيد (metrics, components, balance_index)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    met, comp = {}, {}

    # 1) تماثل القوة
    lat = rot = None
    if _g(m, "str_lat_r") is not None and _g(m, "str_lat_l") is not None:
        lat = met["تماثل القوة — الميل (%)"] = symmetry_index(m["str_lat_r"], m["str_lat_l"])
    if _g(m, "str_rot_r") is not None and _g(m, "str_rot_l") is not None:
        rot = met["تماثل القوة — الدوران (%)"] = symmetry_index(m["str_rot_r"], m["str_rot_l"])
    sym = _mean([lat, rot])
    comp["تماثل القوة"] = linear_score(sym, c["sym_best"], c["sym_worst"]) if sym is not None else None

    # 2) نسبة القوة مثنيات/باسطات
    if _g(m, "str_flex") and _g(m, "str_ext"):
        r = met["نسبة قوة المثنيات/الباسطات"] = m["str_flex"] / m["str_ext"]
        dev = abs(r - c["ref_strength_ratio"]) / c["ref_strength_ratio"]
        comp["نسبة القوة"] = linear_score(dev, c["ratio_best"], c["ratio_worst"])

    # 3) نسبة التحمل
    if _g(m, "end_flex") and _g(m, "end_ext"):
        r = met["نسبة تحمل المثنيات/الباسطات"] = m["end_flex"] / m["end_ext"]
        dev = abs(r - c["ref_endurance_ratio"]) / c["ref_endurance_ratio"]
        comp["نسبة التحمل"] = linear_score(dev, c["ratio_best"], c["ratio_worst"])

    # 4) تماثل PPT
    pa = []
    for a, b, lab in (("ppt_trap_r", "ppt_trap_l", "شبه المنحرفة"), ("ppt_scm_r", "ppt_scm_l", "القصية الترقوية")):
        if _g(m, a) is not None and _g(m, b) is not None:
            v = met[f"تماثل PPT — {lab} (%)"] = symmetry_index(m[a], m[b])
            pa.append(v)
    ppt = _mean(pa)
    comp["تماثل عتبة الألم"] = linear_score(ppt, c["ppt_best"], c["ppt_worst"]) if ppt is not None else None

    # 5) نشاط العضلات السطحية أثناء CCFT (التعويض)
    sup = _mean([_g(m, "emg_scm"), _g(m, "emg_scalene")])
    if sup is not None:
        met["نشاط السطحية أثناء CCFT (% MVC)"] = sup
        comp["ضبط التعويض (EMG)"] = linear_score(sup, c["emg_best"], c["emg_worst"])

    comp = {k: v for k, v in comp.items() if v is not None}
    return met, comp, _mean(list(comp.values()))


def motor_metrics(m: dict, cfg=None):
    c = {**DEFAULT_CFG, **(cfg or {})}
    comp = {}
    for key, lab, b, w in (("mc_track_rmse", "دقة التتبع", "rmse_best", "rmse_worst"),
                           ("mc_sparc", "سلاسة الحركة", "sparc_best", "sparc_worst"),
                           ("mc_coupled", "التحكم الاتجاهي", "coupled_best", "coupled_worst"),
                           ("mc_vel_asym", "تماثل السرعة", "velasym_best", "velasym_worst")):
        v = _g(m, key)
        if v is not None:
            comp[lab] = linear_score(v, c[b], c[w])
    return comp, _mean(list(comp.values()))


def total_index(balance_idx, motor_idx):
    return _mean([balance_idx, motor_idx])
