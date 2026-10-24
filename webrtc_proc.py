# -*- coding: utf-8 -*-
"""معالج فيديو لـ streamlit-webrtc (يُعرَّف عند الطلب لأن المكتبة اختيارية)."""
import threading

import numpy as np
import time

_CLS = None
ICE = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def available():
    try:
        import av  # noqa: F401
        import streamlit_webrtc  # noqa: F401
        return True
    except Exception:
        return False


def get_processor_class():
    global _CLS
    if _CLS is not None:
        return _CLS
    import av
    from streamlit_webrtc import VideoProcessorBase

    from ml.live_coach import LiveSession

    class LiveProc(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.spec = None
            self.mirror = True
            self.session = None
            self.est = None
            self.t0 = None
            self.err = None
            self.face = None
            self.face_err = None
            self.ready = False
            self.n_frames = 0
            self.n_det = 0
            self._fps_t = time.time()
            self._fps_n = 0
            self.fps = 0.0
            threading.Thread(target=self._init_models, daemon=True).start()

        def _init_models(self):
            """تحميل النماذج في خيط خلفي كي لا يتجمد البث أثناء التنزيل/التهيئة."""
            try:
                from ml.neck_pose import PoseEstimator
                est = PoseEstimator()
                try:
                    from ml.neck_pose import HeadPoseEstimator
                    self.face = HeadPoseEstimator()
                except Exception as e:  # الوجه اختياري: نكمل بالجسم فقط
                    self.face_err = str(e)
                with self.lock:
                    self.est = est
                self.ready = True
            except Exception as e:
                self.err = str(e)

        def configure(self, spec, mirror=True):
            with self.lock:
                if spec != self.spec or mirror != self.mirror:
                    self.spec, self.mirror = spec, mirror
                    self.session = None

        def reset(self):
            with self.lock:
                self.session = None

        def recv(self, frame):
            img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            self._fps_n += 1
            if time.time() - self._fps_t >= 2.0:
                self.fps = self._fps_n / (time.time() - self._fps_t)
                self._fps_n, self._fps_t = 0, time.time()
            if not self.ready:  # النموذج قيد التحميل: نُظهر الفيديو فقط
                return av.VideoFrame.from_ndarray(img[:, ::-1].copy() if self.mirror else img, format="rgb24")
            try:
                with self.lock:
                    if self.session is None:
                        self.session = LiveSession(self.est, self.spec, mirror=self.mirror, face=self.face)
                        self.t0 = time.time()
                    img = self.session.process(img, time.time() - self.t0)
                    self.n_frames += 1
                    if self.session.ang is not None:
                        self.n_det += 1
            except Exception as e:  # لا نُسقط البث عند خطأ في النموذج
                self.err = str(e)
                if self.mirror:
                    img = img[:, ::-1].copy()
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        def status(self):
            """نص حالة للعرض: تحميل/خطأ/لا كشف/يعمل + المصدر وعدد الإطارات في الثانية."""
            if self.err:
                return "error", f"تعذّر تشغيل النموذج: {self.err}"
            if not self.ready:
                return "loading", "⏳ جارٍ تحميل نماذج التتبع (أول مرة قد تستغرق نصف دقيقة)…"
            a = self.session.ang if self.session else None
            if a is None:
                return "nodet", "⚠️ لا يُكتشف جسم/وجه: ابتعد عن الكاميرا ليظهر الرأس والكتفان، وحسّن الإضاءة."
            src = "الوجه" if getattr(self.session, "src", "") == "face" else "الجسم"
            return "ok", f"✅ يعمل — المصدر: {src} — {self.fps:.0f} إطار/ث"

        def clock(self):
            return None if self.t0 is None else time.time() - self.t0

        def snapshot(self):
            with self.lock:
                return self.session.last if self.session else None

        def summary(self):
            with self.lock:
                return self.session.coach.summary() if (self.session and self.session.coach) else None

        def timeline(self):
            with self.lock:
                return list(self.session.timeline) if self.session else []

        def current_angles(self):
            with self.lock:
                return self.session.ang if self.session else None

    _CLS = LiveProc
    return _CLS
