# -*- coding: utf-8 -*-
"""
طبقة الأمان (M4): قواعد سريرية صلبة تحدّ من مخرجات الشبكات العصبية
(الجدول 6 في المقترح). هذه الطبقة ليست شبكة عصبية عمدًا: يجب أن تكون
قابلة للمراجعة السريرية ولا يمكن للنموذج تجاوزها.
القيم الرقمية مقترحة وتُعتمد مع المشرف واللجنة الأخلاقية.
"""
from utils.constants import WEEK_RESISTANCE_CAP

PAIN_STOP_ABS = 6.0          # ألم ≥ 6/10
PAIN_STOP_RISE = 2.0         # ارتفاع ≥ نقطتين عن بداية الجلسة
ANGLE_LIMIT_PCT = 90.0       # % من الحد الآمن للزاوية
MAX_STEP_UP = 10.0           # أقصى زيادة في المقاومة بين مجموعتين (نقاط % من MVC)
EXCESS_STREAK = 3            # عدد المجموعات المتتالية «مفرط/تعويضي»
ASYMMETRY_LIMIT = 25.0       # % تفاوت يمين/يسار

SYMPTOMS = ["دوخة", "تنميل", "صداع حاد", "ضعف مفاجئ"]


def evaluate(proposal: dict, state: dict) -> dict:
    """proposal: resistance, reps, sets (مخرجات Dose-Net).
    state: prev_resistance, week, nprs_now, nprs_session_start, head_angle_pct,
           symptoms(list), excess_streak(int), asymmetry_pct(float)."""
    res = float(proposal["resistance"])
    reps, sets = int(proposal["reps"]), int(proposal["sets"])
    prev = float(state.get("prev_resistance", 0.0))
    week = int(state.get("week", 1))
    msgs, action = [], "ok"

    symptoms = state.get("symptoms") or []
    if symptoms:
        return {"action": "stop_session", "resistance": 0.0, "reps": 0, "sets": 0,
                "messages": [f"أعراض خطرة ({'، '.join(symptoms)}): إيقاف الجلسة فورًا وإحالة طبية وتنبيه الأخصائي."],
                "alert_physio": True}

    nprs, start = float(state.get("nprs_now", 0)), float(state.get("nprs_session_start", 0))
    if nprs >= PAIN_STOP_ABS or (nprs - start) >= PAIN_STOP_RISE:
        return {"action": "stop_set", "resistance": 0.0, "reps": 0, "sets": 0,
                "messages": [f"الألم {nprs:g}/10 (بداية الجلسة {start:g}): إيقاف المجموعة وتنبيه الأخصائي."],
                "alert_physio": True}

    if float(state.get("head_angle_pct", 0)) >= ANGLE_LIMIT_PCT:
        res *= 0.8
        action = "reduce"
        msgs.append("الزاوية قريبة من الحد الآمن: خُفضت المقاومة 20% وحُدّد المدى.")

    if int(state.get("excess_streak", 0)) >= EXCESS_STREAK:
        res *= 0.85
        action = "reduce"
        msgs.append("نشاط «مفرط/تعويضي» في 3 مجموعات متتالية: خُفضت المقاومة 15% مع تعليمات تصحيح الوضعية.")

    alert = False
    if float(state.get("asymmetry_pct", 0)) > ASYMMETRY_LIMIT:
        res = min(res, prev)
        alert = True
        action = "reduce" if action == "ok" else action
        msgs.append("تفاوت يمين/يسار > 25%: لا تقدّم في المقاومة وتنبيه للأخصائي.")

    if res > prev + MAX_STEP_UP:
        res = prev + MAX_STEP_UP
        action = "reduce" if action == "ok" else action
        msgs.append(f"قُصّت الزيادة إلى +{MAX_STEP_UP:g} نقاط عن المجموعة السابقة.")

    cap = WEEK_RESISTANCE_CAP.get(week, 70)
    if res > cap:
        res = float(cap)
        action = "reduce" if action == "ok" else action
        msgs.append(f"قُصّت المقاومة إلى سقف الأسبوع {week} ({cap}% من MVC).")

    return {"action": action, "resistance": max(0.0, round(res, 1)),
            "reps": max(8, min(15, reps)), "sets": max(2, min(4, sets)),
            "messages": msgs, "alert_physio": alert}
