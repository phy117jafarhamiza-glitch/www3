# -*- coding: utf-8 -*-
"""مقاييس التحكم الحركي من سلاسل زمنية لزاوية الرأس (من IMU أو الكاميرا)."""
import numpy as np


def velocity(angle, fs):
    a = np.asarray(angle, float)
    return np.gradient(a) * fs  # درجة/ثانية


def peak_velocity(angle, fs):
    v = velocity(angle, fs)
    return float(np.max(np.abs(v))) if len(v) else 0.0


def tracking_rmse(angle, target):
    a, t = np.asarray(angle, float), np.asarray(target, float)
    n = min(len(a), len(t))
    return float(np.sqrt(np.mean((a[:n] - t[:n]) ** 2))) if n else float("nan")


def sparc(angle, fs, fc=10.0, amp_th=0.05, pad=4):
    """Spectral Arc Length (Balasubramanian et al., 2015): قيمة سالبة، الأقرب إلى الصفر = أسلس."""
    speed = np.abs(velocity(angle, fs))
    if len(speed) < 8 or speed.max() == 0:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    mf = mf / mf.max()
    sel = np.where(f <= fc)[0]
    f_sel, mf_sel = f[sel], mf[sel]
    above = np.where(mf_sel >= amp_th)[0]
    if len(above) < 2:
        return 0.0
    lo, hi = above[0], above[-1]
    fs_, ms_ = f_sel[lo:hi + 1], mf_sel[lo:hi + 1]
    df = np.diff(fs_) / (fs_[-1] - fs_[0])
    dm = np.diff(ms_)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def coupled_motion_pct(primary, secondary):
    """الحركة المرافقة: أقصى انحراف في المحور الثانوي كنسبة من أقصى مدى في المحور الأساسي."""
    p, s = np.asarray(primary, float), np.asarray(secondary, float)
    rp = np.ptp(p)
    if rp <= 0:
        return float("nan")
    return float(100.0 * np.max(np.abs(s - s[0])) / rp)


def symmetry_index(a, b):
    """|a-b| / max(a,b) × 100 (%)."""
    a, b = float(a), float(b)
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else 100.0 * abs(a - b) / m
