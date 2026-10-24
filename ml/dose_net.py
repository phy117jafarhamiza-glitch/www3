# -*- coding: utf-8 -*-
"""
Dose-Net (M3): شبكة عصبية أمامية متعددة الطبقات (MLP، طبقتان مخفيتان 32 و16
بتنشيط ReLU) تقترح المقاومة والتكرارات والمجاميع للمجموعة التالية.

التدريب: بالمحاكاة (Imitation Learning) على قرارات وصفها أخصائيون (جدول
dose_expert_labels). عند غياب بيانات الخبراء تُهيَّأ بسياسة خبراء مُرمَّزة
(expert_policy) لأغراض العرض فقط، ويجب استبدالها ببيانات حقيقية قبل التجربة.
لا يُستخدم التعلم المعزز لأنه يتطلب تجريب جرعات على لاعبين مصابين.
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils.constants import DIRECTIONS, DIRECTION_AR, FIXED_DOSE, MODEL_DIR, WEEK_RESISTANCE_CAP

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dosenet.joblib")
FEATURES = ["dir", "quality_normal", "emg_pct_mvc", "head_angle_pct", "velocity", "nprs_now", "week", "prev_resistance"]


def _dir_index(direction):
    if direction in DIRECTIONS:
        return DIRECTIONS.index(direction)
    for i, d in enumerate(DIRECTIONS):
        if DIRECTION_AR[d] == direction:
            return i
    return 0


def _vec(s: dict):
    return [_dir_index(s["direction"]), s["quality_normal"], s["emg_pct_mvc"], s["head_angle_pct"],
            s["velocity"], s["nprs_now"], s["week"], s["prev_resistance"]]


def expert_policy(s: dict) -> dict:
    """سياسة خبراء مُرمَّزة (للتهيئة التجريبية فقط)."""
    wk = int(s["week"])
    cap = WEEK_RESISTANCE_CAP[wk]
    pain_f = float(np.clip(1 - s["nprs_now"] / 8.0, 0.2, 1.0))
    angle_f = float(np.clip(1 - (s["head_angle_pct"] / 100.0) ** 2, 0.25, 1.0))
    vel_f = float(np.clip(1 - abs(s["velocity"] - 40) / 150.0, 0.5, 1.0))
    target = cap * (0.55 + 0.45 * s["quality_normal"]) * pain_f * angle_f * vel_f
    res = min(s["prev_resistance"] + 8.0, target) if target > s["prev_resistance"] else target
    base = FIXED_DOSE[wk]
    reps = int(np.clip(round(base["reps"] + 3 * (s["quality_normal"] - 0.5) - 0.5 * s["nprs_now"]), 8, 15))
    sets = base["sets"] - (1 if s["nprs_now"] >= 4 else 0)
    sets = int(np.clip(sets, 2, 4))
    return {"resistance": float(res), "reps": reps, "sets": sets}


def _synthetic_states(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        wk = int(rng.integers(1, 9))
        rows.append({"direction": DIRECTIONS[int(rng.integers(0, 6))], "quality_normal": float(rng.uniform(0, 1)),
                     "emg_pct_mvc": float(rng.uniform(5, 90)), "head_angle_pct": float(rng.uniform(0, 100)),
                     "velocity": float(rng.uniform(0, 140)), "nprs_now": float(rng.uniform(0, 6)),
                     "week": wk, "prev_resistance": float(rng.uniform(0, WEEK_RESISTANCE_CAP[wk]))})
    return rows


def _fit(states, targets, seed=3):
    X = np.array([_vec(s) for s in states], dtype=float)
    Y = np.array([[t["resistance"] / 100.0, (t["reps"] - 8) / 7.0, (t["sets"] - 2) / 2.0] for t in targets])
    sc = StandardScaler().fit(X)
    net = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", max_iter=1500, random_state=seed,
                       early_stopping=False)
    net.fit(sc.transform(X), Y)
    return sc, net


def train_synthetic():
    states = _synthetic_states()
    sc, net = _fit(states, [expert_policy(s) for s in states])
    meta = {"source": "synthetic", "n": len(states), "trained_at": datetime.now().isoformat(timespec="seconds")}
    joblib.dump({"scaler": sc, "net": net, "meta": meta}, MODEL_PATH)
    return meta


def train_from_expert_labels(rows: list):
    """rows: سجلات dose_expert_labels. يُدمج معها 10% من التهيئة التركيبية لتثبيت الأطراف فقط عند قلة العينات."""
    states, targets = [], []
    for r in rows:
        states.append({"direction": r["direction"], "quality_normal": r["quality_normal"],
                       "emg_pct_mvc": r["emg_pct_mvc"], "head_angle_pct": r["head_angle_pct"],
                       "velocity": r["velocity"], "nprs_now": r["nprs_now"], "week": int(r["week"]),
                       "prev_resistance": r["prev_resistance"]})
        targets.append({"resistance": r["exp_resistance"], "reps": int(r["exp_reps"]), "sets": int(r["exp_sets"])})
    if len(rows) < 300:
        syn = _synthetic_states(n=max(100, len(rows) // 3))
        states += syn
        targets += [expert_policy(s) for s in syn]
    sc, net = _fit(states, targets)
    meta = {"source": "expert", "n": len(rows), "trained_at": datetime.now().isoformat(timespec="seconds")}
    joblib.dump({"scaler": sc, "net": net, "meta": meta}, MODEL_PATH)
    return meta


def evaluate_against_experts(rows: list) -> dict:
    """نسبة الاتفاق مع الأخصائيين: MAE للمقاومة، وتطابق التكرارات والمجاميع."""
    dn = get_dosenet()
    err, rep_ok, set_ok = [], [], []
    for r in rows:
        s = {k: r[k] for k in ["direction", "quality_normal", "emg_pct_mvc", "head_angle_pct", "velocity",
                               "nprs_now", "week", "prev_resistance"]}
        p = dn.predict(s)
        err.append(abs(p["resistance"] - r["exp_resistance"]))
        rep_ok.append(int(p["reps"]) == int(r["exp_reps"]))
        set_ok.append(int(p["sets"]) == int(r["exp_sets"]))
    return {"n": len(rows), "mae_resistance": float(np.mean(err)) if err else None,
            "reps_match": float(np.mean(rep_ok)) if rep_ok else None,
            "sets_match": float(np.mean(set_ok)) if set_ok else None}


class DoseNet:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            train_synthetic()
        b = joblib.load(MODEL_PATH)
        self.scaler, self.net, self.meta = b["scaler"], b["net"], b["meta"]

    @property
    def is_synthetic(self):
        return self.meta.get("source") == "synthetic"

    def predict(self, s: dict) -> dict:
        y = self.net.predict(self.scaler.transform([_vec(s)]))[0]
        return {"resistance": float(np.clip(y[0] * 100.0, 0, 100)),
                "reps": int(np.clip(round(y[1] * 7 + 8), 8, 15)),
                "sets": int(np.clip(round(y[2] * 2 + 2), 2, 4))}


_single = None


def get_dosenet():
    global _single
    if _single is None:
        _single = DoseNet()
    return _single


def reload_dosenet():
    global _single
    _single = DoseNet()
    return _single
