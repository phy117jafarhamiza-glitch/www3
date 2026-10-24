# -*- coding: utf-8 -*-
"""مقاييس الاتفاق لتحقق صلاحية القياس بالكاميرا مقابل المقياس المرجعي (ICC, Bland–Altman, MAE)."""
import numpy as np
from scipy import stats


def icc_2_1(a, b):
    """ICC(2,1): نموذج عشوائي ثنائي الاتجاه، اتفاق مطلق، قياس مفرد (Koo & Li, 2016)
    مع فاصل ثقة 95% تقريبي عبر توزيع F."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, k = len(a), 2
    if n < 3:
        return {"icc": None, "ci": (None, None)}
    Y = np.column_stack([a, b])
    gm = Y.mean()
    ssr = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(axis=0) - gm) ** 2).sum()
    sst = ((Y - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    icc = float((msr - mse) / denom) if denom != 0 else None
    # فاصل الثقة (McGraw & Wong, 1996)
    try:
        F = msr / mse
        Fj = msc / mse
        vn = (k - 1) * (n - 1) * (k * icc * Fj + n * (1 + (k - 1) * icc) - k * icc) ** 2
        vd = (n - 1) * k ** 2 * icc ** 2 * Fj ** 2 + (n * (1 + (k - 1) * icc) - k * icc) ** 2
        dfd = vn / vd
        F95 = stats.f.ppf(0.975, n - 1, dfd)
        F5 = stats.f.ppf(0.975, dfd, n - 1)
        lo = n * (msr - F95 * mse) / (F95 * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (F5 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * F5 * msr)
        ci = (float(lo), float(hi))
    except Exception:
        ci = (None, None)
    return {"icc": icc, "ci": ci}


def bland_altman(cam, ref):
    cam, ref = np.asarray(cam, float), np.asarray(ref, float)
    diff = cam - ref
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mae": float(np.abs(diff).mean()), "n": int(len(diff))}


def interpret_icc(icc):
    if icc is None:
        return "—"
    return "ضعيف" if icc < .5 else "متوسط" if icc < .75 else "جيد" if icc < .9 else "ممتاز"
