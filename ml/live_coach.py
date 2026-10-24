# -*- coding: utf-8 -*-
"""
المدرّب الحركي المباشر: تقدير زوايا الرأس من نقاط BlazePose، ثم آلة حالات لكل تمرين
تعدّ التكرارات وتعطي تغذية راجعة فورية (المدى، السرعة، التعويض، الحد الآمن).

القيود: كاميرا 2D تعطي زوايا تقريبية (خطأ قد يبلغ 5–10°) — يجب إثبات صلاحيتها مقابل مقياس مرجعي.
الثني/البسط/الميل تُقاس بدقة أفضل من الدوران. التمارين الإيزومترية لا تُتتبَّع بالكاميرا.
"""
import math

import numpy as np

NOSE, L_EAR, R_EAR, L_SH, R_SH = 0, 7, 8, 11, 12
AXES = ("pitch", "roll", "yaw")
FRONT_MIN_SHOULDER_W = 0.12   # عرض الكتفين (نسبة من عرض الإطار): أقل منه = لقطة جانبية
VIEW_AR = {"side": "جانبية", "front": "أمامية"}


# ------------------------------------------------------------------ الزوايا
def _norm90(a):
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


def angles_from_landmarks(lm):
    """lm: قائمة (x,y,z) لـ 33 نقطة. يعيد قاموسًا أو None.
    pitch: ثني أمامي موجب (لقطة جانبية). roll: ميل جانبي (لقطة أمامية). yaw: دوران تقريبي (أمامية)."""
    if lm is None or len(lm) < 13:
        return None
    ear = np.mean([lm[L_EAR][:2], lm[R_EAR][:2]], axis=0)
    sh = np.mean([lm[L_SH][:2], lm[R_SH][:2]], axis=0)
    dx, dy = ear[0] - sh[0], ear[1] - sh[1]
    facing = 1.0 if lm[NOSE][0] >= ear[0] else -1.0
    pitch = math.degrees(math.atan2(dx * facing, -dy))
    roll = _norm90(math.degrees(math.atan2(lm[L_EAR][1] - lm[R_EAR][1], lm[L_EAR][0] - lm[R_EAR][0])))
    width = abs(lm[L_SH][0] - lm[R_SH][0]) or 1e-6
    yaw = 70.0 * (lm[NOSE][0] - sh[0]) / width
    tilt = _norm90(math.degrees(math.atan2(lm[L_SH][1] - lm[R_SH][1], lm[L_SH][0] - lm[R_SH][0])))
    view = "front" if width >= FRONT_MIN_SHOULDER_W else "side"
    return {"pitch": pitch, "roll": roll, "yaw": float(yaw), "sh_y": float(sh[1]), "sh_tilt": tilt,
            "facing": facing, "sh_w": float(width), "view": view}


# ---------------------------------------------------------------- المدرب
class ExerciseCoach:
    """spec: dict(mode: rep|hold|neutral_hold, axis: pitch|roll|yaw|multi, sign, target, safe, min_rep_s, hold_s)"""

    def __init__(self, spec, ema=0.4):
        self.spec = dict(spec)
        self.ema = ema
        self.neutral = None
        self._cal = []
        self.reset()

    def reset(self):
        self.state = "calibrating" if self.neutral is None else "neutral"
        self.reps, self.rep_scores = 0, []
        self.too_fast = self.short_rom = self.comp = self.over = 0
        self.max_vel = 0.0
        self.mean_roms = []
        self._f = None
        self._t0 = None
        self._peak = 0.0
        self._rep_flags = set()
        self._prev = None
        self.hold_time = 0.0
        self._last_t = None
        self._dev = False
        self.errors = []
        self.facing = 1.0
        self.messages = []
        self.rel = 0.0
        self.axis_now = self.spec.get("axis", "pitch")

    # ---- المعايرة (الوضع المحايد)
    def calibrate_add(self, ang):
        self._cal.append(ang)

    def calibrate_done(self, min_frames=15):
        if len(self._cal) < min_frames:
            return False
        keys = [k for k, v in self._cal[0].items() if isinstance(v, (int, float))]
        self.neutral = {k: float(np.median([a[k] for a in self._cal])) for k in keys}
        self.state = "neutral"
        return True

    def _rel(self, f):
        sp = self.spec
        n = self.neutral
        ax = sp.get("axis", "pitch")
        if ax == "multi":
            devs = {a: f[a] - n[a] for a in AXES}
            ax = max(devs, key=lambda a: abs(devs[a]))
            self.axis_now = ax
            return abs(devs[ax])
        self.axis_now = ax
        d = f[ax] - n[ax]
        sign = sp.get("sign")
        return d * sign if sign else abs(d)

    # ---- تحديث لكل إطار
    def update(self, t, ang):
        """t بالثواني، ang: ناتج angles_from_landmarks. يعيد قاموس الحالة الحالية."""
        sp = self.spec
        msgs = []
        self.errors = []
        if ang is None:
            self.messages = ["لا يظهر الجسم في الكاميرا"]
            self.errors = list(self.messages)
            return self.snapshot()
        want = sp.get("view")
        if want and ang.get("view") != want:
            self.messages = [f"غيّر زاوية الكاميرا: المطلوب لقطة {VIEW_AR[want]} (الحالية {VIEW_AR.get(ang.get('view'), '؟')})"]
            self.errors = list(self.messages)
            self._last_t = t
            return self.snapshot()
        self.facing = ang.get("facing", 1.0)
        if self.neutral is None:
            self.calibrate_add(ang)
            self.messages = ["قف بوضع محايد ثابت للمعايرة"]
            return self.snapshot()

        num = {k: v for k, v in ang.items() if isinstance(v, (int, float))}
        f = num if self._f is None else {k: self.ema * num[k] + (1 - self.ema) * self._f[k] for k in num}
        self._f = f
        rel = self._rel(f)
        dt = (t - self._last_t) if self._last_t is not None else 0.0
        vel = abs(rel - self.rel) / dt if dt > 1e-3 else 0.0
        self.rel, self._last_t = rel, t
        self.max_vel = max(self.max_vel, vel)
        target, safe = sp["target"], sp["safe"]

        # تعويض: رفع الكتف أو ميلان الجذع
        comp = []
        if self.neutral["sh_y"] - f["sh_y"] > 0.03:
            comp.append("الكتف يرتفع — أرخِه")
        if abs(f["sh_tilt"] - self.neutral["sh_tilt"]) > 6.0:
            comp.append("الجذع يميل — اعتدل")

        if rel > safe:
            msgs.append("توقّف: تجاوزت الحد الآمن")
            self.errors.append(msgs[-1])
        mode = sp.get("mode", "rep")
        sign = sp.get("sign")
        ax = sp.get("axis", "pitch")
        if sign and ax in AXES:
            signed = sign * (f[ax] - self.neutral[ax])
            if signed < -0.3 * target:
                msgs.append("الاتجاه خاطئ — حرّك الرأس في الاتجاه المعاكس")
                self.errors.append(msgs[-1])

        if mode == "rep":
            max_speed = 1.5 * 2 * target / sp.get("min_rep_s", 3.0)
            if vel > max_speed and self.state != "neutral":
                msgs.append("أبطئ الحركة")
                self.errors.append(msgs[-1])
            if self.state == "neutral" and rel > 0.25 * target:
                self.state, self._t0, self._peak, self._rep_flags = "moving", t, rel, set()
            elif self.state == "moving":
                self._peak = max(self._peak, rel)
                if comp:
                    self._rep_flags.add("comp")
                if rel > safe:
                    self._rep_flags.add("over")
                if vel > max_speed:
                    self._rep_flags.add("fast")
                if self._peak < 0.8 * target and rel < self._peak - 0.1 * target:
                    msgs.append("اذهب لمدى أكبر")
                    self.errors.append(msgs[-1])
                elif self._peak >= 0.8 * target and rel >= 0.8 * target:
                    msgs.append("جيد — ثبّت لحظة ثم ارجع ببطء")
                if rel < 0.25 * target:
                    self._finish_rep(t)
        elif mode == "hold":
            in_zone = 0.8 * target <= rel <= safe
            if in_zone and dt:
                self.hold_time += dt
            msgs.append("ثبّت الوضع" if in_zone else "ادخل المنطقة الخضراء")
            if self.hold_time >= sp.get("hold_s", 20):
                msgs.append("أُنجز زمن الثبات")
        elif mode == "neutral_hold":
            if rel <= target and dt:
                self.hold_time += dt
                self._dev = False
                msgs.append("ثبات ممتاز")
            else:
                msgs.append("الرأس ينحرف — أعده للوضع المحايد")
                if dt and not self._dev:
                    self.over += 1
                self._dev = True
        if comp:
            msgs.extend(comp)
            self.errors.extend(comp)
        if mode == "neutral_hold" and rel > target:
            self.errors.append("الرأس ينحرف")
        self.messages = msgs or ["جاهز — ابدأ الحركة"]
        return self.snapshot()

    def _finish_rep(self, t):
        dur = t - self._t0
        flags = set(self._rep_flags)
        sp = self.spec
        if self._peak >= 0.5 * sp["target"]:
            self.reps += 1
            score = 100.0
            if self._peak < 0.8 * sp["target"]:
                self.short_rom += 1
                score -= 25
            if dur < sp.get("min_rep_s", 3.0) or "fast" in flags:
                self.too_fast += 1
                score -= 25
            if "comp" in flags:
                self.comp += 1
                score -= 30
            if "over" in flags:
                self.over += 1
                score -= 20
            self.rep_scores.append(max(0.0, score))
            self.mean_roms.append(self._peak)
        self.state = "neutral"

    def snapshot(self):
        sp = self.spec
        return {"state": self.state, "reps": self.reps, "rel": self.rel, "axis": self.axis_now,
                "messages": list(self.messages), "errors": list(self.errors), "error": bool(self.errors),
                "arrow": self._arrow(), "target": sp["target"], "safe": sp["safe"],
                "hold_time": self.hold_time, "hold_goal": sp.get("hold_s"), "mode": sp.get("mode", "rep")}

    def _arrow(self):
        """اتجاه التصحيح المطلوب في إحداثيات الصورة الخام (dx, dy) أو None."""
        sp = self.spec
        ax, sign = sp.get("axis"), sp.get("sign")
        if not sign or ax not in AXES:
            return None
        if ax == "pitch":      # +1 ثني أمامي: نحو اتجاه النظر وللأسفل ؛ -1 بسط: للخلف وللأعلى
            return (self.facing * 0.7 * (1 if sign > 0 else -1), 0.7 if sign > 0 else -0.7)
        # roll/yaw: -1 = يمين الشخص = يسار صورة الكاميرا غير المعكوسة
        return (-1.0 if sign < 0 else 1.0, 0.25 if ax == "roll" else 0.0)

    def summary(self):
        if self.spec.get("mode", "rep") == "rep":
            q = float(np.mean(self.rep_scores)) if self.rep_scores else 0.0
        else:
            goal = self.spec.get("hold_s") or 1.0
            q = max(0.0, min(100.0, 100.0 * self.hold_time / goal) - 10.0 * self.over)
        return {"reps": self.reps, "mean_rom": float(np.mean(self.mean_roms)) if self.mean_roms else 0.0,
                "max_velocity": float(self.max_vel), "too_fast": self.too_fast, "short_rom": self.short_rom,
                "compensation": self.comp, "over_limit": self.over, "quality_score": q,
                "hold_time": self.hold_time}


# ---------------------------------------------------------------- جلسة الإطارات
class LiveSession:
    """يربط مقدِّر الوضعية بالمدرّب (اختياري) ويسجّل السلسلة الزمنية للزوايا، ويرسم تراكبًا على الإطار.
    التحليل يتم على الصورة الخام دائمًا؛ التعكيس (mirror) للعرض فقط. cv2 لا يرسم العربية، فالتراكب أرقام
    وأشرطة وأسهم وإطار أحمر عند الخطأ، والرسائل العربية تُعرض في الواجهة."""

    FACE_MAX_YAW = 60.0

    def __init__(self, estimator, spec=None, mirror=False, record=True, calib_frames=30, face=None):
        self.est = estimator
        self.face = face
        self.src = "pose"
        self.coach = ExerciseCoach(spec) if spec else None
        self.mirror = mirror
        self.record = record
        self.calib_frames = calib_frames
        self.timeline = []
        self.last = self.coach.snapshot() if self.coach else {"state": "free", "reps": 0, "rel": 0.0, "messages": [],
                                                              "errors": [], "error": False, "arrow": None,
                                                              "target": 1, "safe": 2}
        self.ang = None

    def process(self, rgb, t):
        lm = self.est.estimate(rgb)
        ang = angles_from_landmarks(lm) if lm else None
        self.src = "pose"
        if self.face is not None:
            try:
                fa = self.face.estimate(rgb)
            except Exception:
                fa = None
            # الوجه أدق في اللقطة الأمامية (دوران/ميل/ثني)، ويُهمَل عند الدوران الكبير أو اللقطة الجانبية
            if fa and abs(fa["yaw"]) <= self.FACE_MAX_YAW and (ang is None or ang["view"] == "front"):
                base = ang or {"sh_y": 0.5, "sh_tilt": 0.0, "facing": 1.0, "sh_w": 0.3, "view": "front"}
                ang = {**base, "pitch": fa["pitch"], "yaw": fa["yaw"], "roll": fa["roll"], "view": "front"}
                self.src = "face"
        self.ang = ang
        if self.record and ang:
            self.timeline.append({"t": float(t), **ang})
        if self.coach:
            snap = self.coach.update(t, ang)
            if self.coach.neutral is None and len(self.coach._cal) >= self.calib_frames:
                self.coach.calibrate_done()
                snap = self.coach.snapshot()
            self.last = snap
        return self.draw(rgb, lm, self.last)

    def draw(self, rgb, lm, snap):
        import cv2
        img = np.ascontiguousarray(rgb.copy())
        h, w = img.shape[:2]
        if lm:
            lm = [(p[0], p[1], p[2]) for p in lm]
        if self.mirror:
            img = np.ascontiguousarray(img[:, ::-1])
            if lm:
                lm = [(1.0 - p[0], p[1], p[2]) for p in lm]
        if lm:
            for i in (NOSE, L_EAR, R_EAR, L_SH, R_SH):
                cv2.circle(img, (int(lm[i][0] * w), int(lm[i][1] * h)), 5, (0, 200, 255), -1)
            cv2.line(img, (int(lm[L_SH][0] * w), int(lm[L_SH][1] * h)), (int(lm[R_SH][0] * w), int(lm[R_SH][1] * h)), (255, 200, 0), 2)
        if not self.coach:
            return img
        # أخطاء: إطار أحمر + سهم يوضّح اتجاه التصحيح
        if snap.get("error"):
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 40, 40), 8)
            arr = snap.get("arrow")
            if arr and lm:
                dx, dy = arr
                if self.mirror:
                    dx = -dx
                x0, y0 = int(lm[NOSE][0] * w), int(lm[NOSE][1] * h)
                cv2.arrowedLine(img, (x0, y0), (int(x0 + dx * 0.18 * w), int(y0 + dy * 0.18 * w)), (255, 60, 60), 5, tipLength=0.35)
        elif snap.get("state") != "calibrating":
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 200, 90), 4)
        # شريط المدى: أخضر = المنطقة المستهدفة، أحمر = ما بعد الحد الآمن
        x0, y0, bw, bh = 20, h - 40, w - 40, 16
        mx = snap["safe"] * 1.25
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), -1)
        gx = lambda v: int(x0 + bw * min(1.0, max(0.0, v / mx)))
        cv2.rectangle(img, (gx(0.8 * snap["target"]), y0), (gx(snap["safe"]), y0 + bh), (60, 180, 75), -1)
        cv2.rectangle(img, (gx(snap["safe"]), y0), (x0 + bw, y0 + bh), (60, 60, 220), -1)
        cv2.circle(img, (gx(snap["rel"]), y0 + bh // 2), 9, (255, 255, 255), -1)
        cv2.putText(img, f"REPS {snap['reps']}  {snap['rel']:.0f} / {snap['target']:.0f} deg", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if snap["state"] == "calibrating":
            cv2.putText(img, "CALIBRATING - hold neutral", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        return img


def process_video(path, spec, estimator, stride=2, max_seconds=180, progress=None):
    """معالجة فيديو مسجَّل بالمدرّب نفسه. يعيد (summary, last_annotated_frame, timeline[list of dict])."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ls = LiveSession(estimator, spec)
    i, last, timeline = 0, None, []
    while True:
        ok, bgr = cap.read()
        if not ok or i / fps > max_seconds:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last = ls.process(rgb, i / fps)
            timeline.append({"t": i / fps, "angle": ls.last["rel"], "reps": ls.last["reps"]})
            if progress and total:
                progress(min(1.0, i / total))
        i += 1
    cap.release()
    return (ls.coach.summary() if ls.coach else None), last, timeline


def track_video(path, estimator, stride=2, max_seconds=240, progress=None):
    """تتبّع فيديو دون مدرّب: يعيد قائمة dict(t, pitch, roll, yaw, sh_y, sh_tilt, facing, sh_w, view)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ls = LiveSession(estimator, None, record=True)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or i / fps > max_seconds:
            break
        if i % stride == 0:
            ls.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i / fps)
            if progress and total:
                progress(min(1.0, i / total))
        i += 1
    cap.release()
    return ls.timeline
