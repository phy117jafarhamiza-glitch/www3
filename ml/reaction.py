# -*- coding: utf-8 -*-
"""اختبار سرعة الاستجابة الحركية للرأس: يظهر مثير (سهم) وتُقاس بداية حركة الرأس في الاتجاه الصحيح.

القيود: زمن الاستجابة المقاس يتضمن تأخر الكاميرا والشبكة والعرض (قد يبلغ 100–300 ms)؛ لذلك يصلح للمقارنة
بين T0 وT3 لنفس اللاعب بنفس الجهاز أكثر من الأرقام المطلقة.
اتجاهات المثير (بالنسبة للاعب): دوران يمين/يسار (yaw) وميل يمين/يسار (roll) من لقطة أمامية؛ ثني/بسط (pitch) من لقطة جانبية.
"""
import numpy as np

STIMULI = {
    "turn_r": dict(ar="دوران يمين", axis="yaw", sign=-1, view="front", arrow="⬅️" ),
    "turn_l": dict(ar="دوران يسار", axis="yaw", sign=+1, view="front", arrow="➡️"),
    "tilt_r": dict(ar="ميل يمين", axis="roll", sign=-1, view="front", arrow="↙️"),
    "tilt_l": dict(ar="ميل يسار", axis="roll", sign=+1, view="front", arrow="↘️"),
    "flex": dict(ar="ثني أمامي", axis="pitch", sign=+1, view="side", arrow="⬇️"),
    "ext": dict(ar="بسط خلفي", axis="pitch", sign=-1, view="side", arrow="⬆️"),
}
# ملاحظة العرض: الشاشة تُعرض معكوسة (مرآة)، فالسهم يشير إلى جهة الشاشة؛ لذا يُرسم السهم من منظور اللاعب:
# "يمين اللاعب" = يمين الشاشة المعكوسة. نستخدم نصًا واضحًا مع السهم لتفادي اللبس.
SCREEN_ARROW = {"turn_r": "➡️", "turn_l": "⬅️", "tilt_r": "↘️", "tilt_l": "↙️", "flex": "⬇️", "ext": "⬆️"}


def analyze_trial(tl, t_stim, axis, sign, window=2.5, thr=4.0, base_s=0.3, min_latency=0.08, peak_frac=0.9):
    """tl: قائمة dict(t, pitch, roll, yaw...). يعيد dict(rt_ms, mt_ms, peak_velocity, correct, peak_deg)."""
    t = np.array([r["t"] for r in tl], float)
    x = np.array([r[axis] for r in tl], float)
    pre = (t >= t_stim - base_s) & (t <= t_stim)
    if pre.sum() < 2:
        return {"rt_ms": None, "mt_ms": None, "peak_velocity": None, "correct": False, "peak_deg": None, "note": "لا إطارات قبل المثير"}
    base = float(np.median(x[pre]))
    post = (t > t_stim + min_latency) & (t <= t_stim + window)
    if post.sum() < 3:
        return {"rt_ms": None, "mt_ms": None, "peak_velocity": None, "correct": False, "peak_deg": None, "note": "لا إطارات بعد المثير"}
    tp, dev = t[post], sign * (x[post] - base)
    ok_idx = np.where(dev > thr)[0]
    bad_idx = np.where(-dev > thr)[0]
    if len(ok_idx) == 0:
        return {"rt_ms": None, "mt_ms": None, "peak_velocity": None, "correct": False, "peak_deg": float(dev.max()),
                "note": "لم تُرصد استجابة" if len(bad_idx) == 0 else "اتجاه خاطئ"}
    i_ok = ok_idx[0]
    wrong_first = len(bad_idx) > 0 and bad_idx[0] < i_ok
    onset = float(tp[i_ok])
    seg = dev[i_ok:]
    peak = float(seg.max())
    j = int(np.argmax(seg >= peak_frac * peak))
    mt = float(tp[i_ok + j] - onset) * 1000.0
    vel = np.abs(np.gradient(dev, tp))
    return {"rt_ms": (onset - t_stim) * 1000.0, "mt_ms": mt, "peak_velocity": float(vel.max()),
            "correct": not wrong_first, "peak_deg": peak, "note": "اتجاه خاطئ أولًا" if wrong_first else ""}


def summarize(trials):
    ok = [x for x in trials if x.get("correct") and x.get("rt_ms") is not None]
    out = {"n": len(trials), "n_correct": len(ok),
           "rt_acc": 100.0 * len(ok) / len(trials) if trials else None}
    if ok:
        rts = np.array([x["rt_ms"] for x in ok])
        out.update({"rt_ms": float(np.mean(rts)), "rt_median": float(np.median(rts)), "rt_sd": float(rts.std(ddof=1)) if len(rts) > 1 else 0.0,
                    "mt_ms": float(np.mean([x["mt_ms"] for x in ok])),
                    "peak_velocity": float(np.mean([x["peak_velocity"] for x in ok]))})
    return out
