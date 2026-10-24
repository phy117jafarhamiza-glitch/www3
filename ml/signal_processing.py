# -*- coding: utf-8 -*-
"""
معالجة إشارة EMG الخام وتحويلها إلى ميزات (Features) صالحة للتصنيف بالشبكة
العصبية، وفق ما ورد في الأطروحة (تخطيط كهربية العضلات - EMG, De Luca 1997).

خط المعالجة:
1) إزالة الانحياز (DC offset removal)
2) مرشح تمرير نطاقي Bandpass (20-450Hz تقريبياً لإشارة EMG سطحية)
3) التقويم الموجي الكامل Full-wave rectification
4) غلاف RMS متحرك (moving RMS envelope)
5) تقسيم الإشارة الى نوافذ (windowing) واستخراج ميزات كل نافذة
"""
import numpy as np

try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False


def bandpass_filter(x: np.ndarray, fs: float, low=20.0, high=450.0, order=4) -> np.ndarray:
    """مرشح تمرير نطاقي لإشارة EMG الخام. يتراجع بأمان اذا لم تتوفر scipy
    أو اذا كان معدل العينات fs غير كافٍ لحدود الترشيح."""
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    high = min(high, nyq * 0.98)
    if not SCIPY_AVAILABLE or high <= low:
        # تراجع بسيط: إزالة الانحياز فقط
        return x - np.mean(x)
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, x - np.mean(x))


def notch_filter(x: np.ndarray, fs: float, f0=50.0, q=30.0) -> np.ndarray:
    """مرشح قاطع لتردد التيار الكهربائي (50 Hz في العراق)."""
    if not SCIPY_AVAILABLE or f0 >= fs / 2.0:
        return np.asarray(x, dtype=float)
    b, a = sp_signal.iirnotch(f0 / (fs / 2.0), q)
    return sp_signal.filtfilt(b, a, np.asarray(x, dtype=float))


def rms_envelope(x: np.ndarray, fs: float, window_ms=50.0) -> np.ndarray:
    """غلاف RMS متحرك بحجم نافذة window_ms مللي ثانية."""
    win = max(1, int(fs * window_ms / 1000.0))
    x2 = np.asarray(x, dtype=float) ** 2
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(x2, kernel, mode="same"))


def segment_signal(x: np.ndarray, fs: float, window_s=0.5, overlap=0.5):
    """يقسم الإشارة الى نوافذ متراكبة (overlapping windows) لاستخراج الميزات."""
    win = int(fs * window_s)
    step = max(1, int(win * (1 - overlap)))
    segments = []
    for start in range(0, max(1, len(x) - win + 1), step):
        seg = x[start:start + win]
        if len(seg) == win:
            segments.append(seg)
    if not segments and len(x) > 0:
        segments = [x]
    return segments


def extract_features(segment: np.ndarray) -> dict:
    """ميزات EMG الكلاسيكية المستخدمة في الأدبيات (De Luca 1997؛ Phinyomark et al.):
    RMS, MAV, Zero Crossings, Waveform Length, Variance, Slope Sign Changes.
    هذه الميزات هي مدخلات الشبكة العصبية / المصنّف.
    """
    seg = np.asarray(segment, dtype=float)
    if len(seg) == 0:
        return {"rms": 0.0, "mav": 0.0, "zc": 0.0, "wl": 0.0, "var": 0.0, "ssc": 0.0}

    rms = float(np.sqrt(np.mean(seg ** 2)))
    mav = float(np.mean(np.abs(seg)))
    zc = float(np.sum(np.diff(np.sign(seg)) != 0))
    wl = float(np.sum(np.abs(np.diff(seg))))
    var = float(np.var(seg))
    diffs = np.diff(seg)
    ssc = float(np.sum(np.diff(np.sign(diffs)) != 0)) if len(diffs) > 1 else 0.0
    return {"rms": rms, "mav": mav, "zc": zc, "wl": wl, "var": var, "ssc": ssc}


def features_matrix(segments):
    """يحول قائمة نوافذ الى مصفوفة ميزات (N x 6) جاهزة للنموذج."""
    feats = [extract_features(s) for s in segments]
    keys = ["rms", "mav", "zc", "wl", "var", "ssc"]
    return np.array([[f[k] for k in keys] for f in feats]), keys


def raw_windows_matrix(segments, target_len=None):
    """يحول قائمة نوافذ الى مصفوفة (N x L) للإشارة الخام، صالحة كمدخل CNN 1D
    (بعد إعادة تحجيم كل نافذة الى نفس الطول target_len)."""
    if not segments:
        return np.zeros((0, target_len or 1))
    if target_len is None:
        target_len = int(np.median([len(s) for s in segments]))
    out = []
    for s in segments:
        if len(s) == target_len:
            out.append(s)
        else:
            # إعادة أخذ العينات (linear resample) الى الطول الموحّد
            x_old = np.linspace(0, 1, len(s))
            x_new = np.linspace(0, 1, target_len)
            out.append(np.interp(x_new, x_old, s))
    return np.array(out)


def full_pipeline(raw_signal, fs, window_s=0.5, overlap=0.5, target_len=256, ref_amp=None, notch=True):
    """خط المعالجة الكامل. يعيد الإشارة المرشّحة، غلاف RMS، النوافذ الخام،
    مصفوفة الميزات، والنوافذ المطبَّعة (raw_windows) الجاهزة لـ EMG-Net.

    ref_amp: اتساع مرجعي (RMS عند أقصى انقباض إرادي MVC) لتطبيع النوافذ؛
    اذا لم يُعطَ يُستخدم المئين 95 لغلاف RMS كتقدير تقريبي."""
    filtered = bandpass_filter(raw_signal, fs)
    if notch:
        filtered = notch_filter(filtered, fs)
    envelope = rms_envelope(filtered, fs)
    if ref_amp is None or ref_amp <= 0:
        ref_amp = float(np.percentile(envelope, 95)) or 1.0
    segments = segment_signal(filtered, fs, window_s, overlap)
    feats, feat_keys = features_matrix(segments)
    raw_win = raw_windows_matrix(segments, target_len=target_len) / ref_amp
    return {
        "filtered": filtered, "envelope": envelope, "segments": segments,
        "features": feats, "feature_keys": feat_keys, "raw_windows": raw_win, "ref_amp": ref_amp,
    }
