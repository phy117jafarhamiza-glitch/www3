# -*- coding: utf-8 -*-
"""
EMG-Net: شبكة عصبية التفافية أحادية البعد (1D-CNN) لتصنيف جودة تنشيط عضلات
الرقبة من نافذة EMG (0.5 ث) إلى ثلاث فئات: ضعيف/غير متوازن، طبيعي، مفرط/تعويضي.

النوع: تعلّم عميق خاضع للإشراف (Supervised Deep Learning)، مبني بـ NumPy
(انظر ml/numpy_cnn.py) ليعمل في أي بيئة ويمكن تدريبه داخل التطبيق.

تنبيه علمي: النموذج الأولي مُدرَّب على بيانات تركيبية للعرض والاختبار فقط.
لا يجوز الاستشهاد بدقته في الأطروحة. يجب تدريبه على نوافذ EMG حقيقية موسومة
من خبيرين (صفحة «تدريب والتحقق من النماذج») مع تقسيم بحسب اللاعب.
"""
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from ml.numpy_cnn import EMGNet1DCNN
from ml.signal_processing import features_matrix
from utils.constants import EMG_QUALITY_LABELS, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "emgnet.npz")
META_PATH = os.path.join(MODEL_DIR, "emgnet_meta.json")
INPUT_LEN = 256
FEATURE_KEYS = ["rms", "mav", "zc", "wl", "var", "ssc"]


# --------------------------------------------------------------------------
def synthetic_dataset(n_subjects=18, n_per_class_per_subject=14, seed=42):
    """بيانات تركيبية بثلاثة أنماط، مع تباين فردي بين «اللاعبين» (groups)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 0.5, INPUT_LEN, endpoint=False)
    profiles = {0: (0.15, 0.08, 60), 1: (0.45, 0.05, 90), 2: (0.85, 0.12, 140)}
    X, y, g = [], [], []
    for s in range(n_subjects):
        gain = 1 + 0.2 * rng.standard_normal()
        for lab, (amp, noise, freq) in profiles.items():
            for _ in range(n_per_class_per_subject):
                sig = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                sig = (sig + noise * rng.standard_normal(INPUT_LEN)) * gain
                X.append(sig); y.append(lab); g.append(s)
    return np.array(X), np.array(y), np.array(g)


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def cross_validate_subjectwise(X, y, groups, n_splits=5, epochs=20, seed=0):
    """تحقق متقاطع بحسب اللاعب (لا تتسرب نوافذ اللاعب نفسه بين التدريب والاختبار)."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    preds = np.zeros_like(y)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        net = EMGNet1DCNN(INPUT_LEN, 3, seed=seed + k)
        net.fit(X[tr], y[tr], epochs=epochs, seed=seed + k)
        preds[te] = net.predict(X[te])
        folds.append(float(accuracy_score(y[te], preds[te])))
    res = _metrics(y, preds)
    res["fold_accuracy"] = folds
    res["n_windows"] = int(len(y))
    res["n_subjects"] = int(len(np.unique(groups)))
    return res


def train_and_save(X, y, epochs=25, source="synthetic", extra_meta=None, seed=0):
    net = EMGNet1DCNN(INPUT_LEN, 3, seed=seed)
    hist = net.fit(X, y, epochs=epochs, seed=seed)
    net.save(MODEL_PATH)
    meta = {"source": source, "n_windows": int(len(y)), "trained_at": datetime.now().isoformat(timespec="seconds"),
            "final_loss": hist[-1]["loss"], "class_counts": np.bincount(y, minlength=3).tolist()}
    meta.update(extra_meta or {})
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return net, meta


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


class EMGClassifier:
    """الواجهة المستخدمة في التطبيق."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            X, y, _ = synthetic_dataset()
            train_and_save(X, y, epochs=20, source="synthetic")
        self.net = EMGNet1DCNN.load(MODEL_PATH)
        self.meta = load_meta()

    @property
    def backend(self):
        src = self.meta.get("source", "synthetic")
        tag = "بيانات تركيبية (للعرض فقط)" if src == "synthetic" else f"بيانات حقيقية موسومة ({self.meta.get('n_windows','?')} نافذة)"
        return f"EMG-Net (1D-CNN) — {tag}"

    @property
    def is_synthetic(self):
        return self.meta.get("source", "synthetic") == "synthetic"

    def predict_windows(self, norm_windows, segments=None):
        """norm_windows: (N,256) مطبَّعة بـ MVC. segments: النوافذ الخام لحساب الميزات (اختياري)."""
        norm_windows = np.asarray(norm_windows, dtype=float)
        if len(norm_windows) == 0:
            return [], None, 0.0
        probs = self.net.predict_proba(norm_windows)
        feats = None
        if segments is not None:
            fm, _ = features_matrix(segments)
            feats = [dict(zip(FEATURE_KEYS, row)) for row in fm]
        results = []
        for i, p in enumerate(probs):
            k = int(np.argmax(p))
            results.append({"label": EMG_QUALITY_LABELS[k], "confidence": float(p[k]),
                            "probs": p.tolist(), "features": feats[i] if feats else {}})
        labels = [r["label"] for r in results]
        majority = max(set(labels), key=labels.count)
        quality_normal = float(probs[:, 1].mean())
        return results, majority, quality_normal


_singleton = None


def get_classifier():
    global _singleton
    if _singleton is None:
        _singleton = EMGClassifier()
    return _singleton


def reload_classifier():
    global _singleton
    _singleton = EMGClassifier()
    return _singleton
