# -*- coding: utf-8 -*-
"""مدى الحركة ومؤشر الحذر الحركي من سلسلة زوايا الرأس (فيديو مسجَّل أو بث مباشر).

الاتفاقيات: pitch + = ثني أمامي؛ roll − = ميل إلى يمين الشخص؛ yaw − = دوران إلى يمين الشخص.
الدوران تقريبي جدًا (إزاحة الأنف نسبة لعرض الكتفين). مؤشر الحذر الحركي سلوكي تجريبي لم تثبت صلاحيته
مقابل استبيان TSK-11، فيُقارَن به ولا يحلّ محلّه.
"""
import numpy as np

FRONT_MIN_SHOULDER_W = 0.12
NORMS = {"flex": 50.0, "ext": 60.0, "lat_r": 45.0, "lat_l": 45.0, "rot_r": 80.0, "rot_l": 80.0}
GUARD_CFG = {"ref_speed": 45.0, "pause_speed": 2.0, "pause_dev": 5.0, "pause_best": 0.0, "pause_worst": 0.5,
             "sh_thr": 0.03}


def _arr(tl, k):
    return np.array([r[k] for r in tl], float)


def _medfilt(x, k=5):
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def detect_view(tl):
    if not tl:
        return None
    w = float(np.median(_arr(tl, "sh_w")))
    return "front" if w >= FRONT_MIN_SHOULDER_W else "side"


def neutral_of(tl, seconds=1.5):
    t = _arr(tl, "t")
    sel = t <= t[0] + seconds
    return {k: float(np.median(_arr(tl, k)[sel])) for k in ("pitch", "roll", "yaw", "sh_y", "sh_tilt")}


def rom_from_timeline(tl, view=None, neutral_s=1.5):
    """يعيد dict(view, rom={dir: درجة}, notes). أول neutral_s ثانية = الوضع المحايد (يجب أن يبدأ اللاعب ثابتًا)."""
    if not tl or len(tl) < 15:
        return {"view": None, "rom": {}, "notes": ["عدد الإطارات غير كافٍ."]}
    view = view or detect_view(tl)
    n = neutral_of(tl, neutral_s)
    rom, notes = {}, []
    if view == "side":
        d = _medfilt(_arr(tl, "pitch") - n["pitch"])
        rom["flex"] = max(0.0, float(np.percentile(d, 99)))
        rom["ext"] = max(0.0, float(-np.percentile(d, 1)))
        notes.append("لقطة جانبية: قيس الثني والبسط فقط.")
    else:
        dr = _medfilt(_arr(tl, "roll") - n["roll"])
        dy = _medfilt(_arr(tl, "yaw") - n["yaw"])
        rom["lat_r"] = max(0.0, float(-np.percentile(dr, 1)))
        rom["lat_l"] = max(0.0, float(np.percentile(dr, 99)))
        rom["rot_r"] = max(0.0, float(-np.percentile(dy, 1)))
        rom["rot_l"] = max(0.0, float(np.percentile(dy, 99)))
        notes.append("لقطة أمامية: قيس الميل والدوران فقط؛ الدوران تقريبي.")
    return {"view": view, "rom": rom, "notes": notes}


def _score(x, best, worst):
    return float(min(1.0, max(0.0, (x - best) / (worst - best))))


def guarding_index(tl, rom=None, view=None, cfg=None):
    """مؤشر الحذر الحركي 0–100 (الأعلى = حذر أكبر). يعيد dict(index, components, details)."""
    c = {**GUARD_CFG, **(cfg or {})}
    if not tl or len(tl) < 15:
        return {"index": None, "components": {}, "details": {}}
    view = view or detect_view(tl)
    n = neutral_of(tl)
    t = _arr(tl, "t")
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 0.03
    if view == "side":
        dev = _medfilt(_arr(tl, "pitch") - n["pitch"])
    else:
        dev = np.hypot(_medfilt(_arr(tl, "roll") - n["roll"]), _medfilt(_arr(tl, "yaw") - n["yaw"]))
    speed = np.abs(np.gradient(dev) / dt)
    comp, det = {}, {}
    v90 = float(np.percentile(speed, 90))
    det["السرعة العليا (90%) °/ث"] = v90
    comp["بطء الحركة"] = 1.0 - min(1.0, v90 / c["ref_speed"])
    active = np.where(np.abs(dev) > c["pause_dev"])[0]
    if len(active) > 5:
        w = slice(active[0], active[-1] + 1)
        pf = float(np.mean((speed[w] < c["pause_speed"]) & (np.abs(dev[w]) > c["pause_dev"])))
        det["نسبة التوقف في منتصف الحركة"] = pf
        comp["التردد والتوقف"] = _score(pf, c["pause_best"], c["pause_worst"])
    if rom:
        defs = [max(0.0, 1.0 - rom[k] / NORMS[k]) for k in rom if k in NORMS]
        if defs:
            det["متوسط نقص المدى عن المرجع"] = float(np.mean(defs))
            comp["تقييد المدى"] = float(np.mean(defs))
    sh = _arr(tl, "sh_y")
    frac = float(np.mean((n["sh_y"] - sh) > c["sh_thr"]))
    det["نسبة رفع الكتف"] = frac
    comp["رفع الكتف"] = min(1.0, frac / 0.3)
    idx = 100.0 * float(np.mean(list(comp.values())))
    return {"index": idx, "components": {k: 100.0 * v for k, v in comp.items()}, "details": det}
