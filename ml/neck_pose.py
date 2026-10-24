# -*- coding: utf-8 -*-
"""
تقدير مدى حركة الرقبة (ROM) عبر الكاميرا باستخدام MediaPipe Pose، كبديل
اقتصادي وسهل الوصول لمقياس الزوايا الرقمي (Digital Inclinometer) ومستشعر
IMU المذكورين في الفصل الثالث من الأطروحة، عند عدم توفر الجهاز أو لإجراء
متابعة سريعة بين الجلسات.

طريقة العمل تحاكي بروتوكول الأطروحة نفسه (تصفير الجهاز ثم قياس الانحراف):
  1) التقاط وضعية محايدة (Neutral) - الرأس في وضع الصفر.
  2) التقاط وضعية الذروة بعد تنفيذ الحركة (ثني/انبساط/ميل/دوران).
  3) حساب الفرق الزاوي بين الوضعيتين = مدى الحركة لتلك الجهة.

ملاحظة منهجية مهمة: الثني والانبساط (Flexion/Extension) يحتاجان الى لقطة
من الجانب (profile view)، بينما الميل الجانبي والدوران يمكن تقديرهما من
لقطة أمامية (frontal view). هذا التقسيم يجب توضيحه للمستخدم في الواجهة.
"""
import os
import numpy as np

MODEL_DIR = "data/models"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# فهارس نقاط مفصل MediaPipe Pose (33 نقطة) ذات الصلة بالرقبة
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def ensure_model_downloaded():
    """يحاول تنزيل نموذج PoseLandmarker عند أول استخدام (يتطلب اتصال
    انترنت على جهاز التشغيل الفعلي). في حال الفشل يرفع استثناء برسالة
    واضحة تشرح كيفية تنزيل الملف يدوياً."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(POSE_MODEL_PATH):
        return POSE_MODEL_PATH
    try:
        import urllib.request
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        return POSE_MODEL_PATH
    except Exception as e:
        raise RuntimeError(
            "تعذّر تنزيل نموذج تتبع الجسم (PoseLandmarker) تلقائياً.\n"
            f"السبب: {e}\n"
            "الحل: نزّل الملف يدوياً من الرابط التالي وضعه في المسار "
            f"'{POSE_MODEL_PATH}':\n{POSE_MODEL_URL}"
        )


class PoseEstimator:
    """غلاف حول MediaPipe Tasks PoseLandmarker لاستخراج إحداثيات نقاط
    الجسم (33 نقطة) من صورة واحدة (إطار فيديو أو صورة ملتقطة)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_model_downloaded()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._mp = mp
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, image_rgb: np.ndarray):
        """image_rgb: مصفوفة numpy بصيغة RGB (H,W,3). يعيد قائمة نقاط
        (x, y, z) طبيعية [0..1] أو None اذا لم يتم اكتشاف جسم."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        return [(p.x, p.y, p.z) for p in lm]


# ---------------------------------------------------------------- الهندسة
def _angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def head_pitch_vector(landmarks):
    """متجه (كتف->أذن) في مستوى (x,y) - يُستخدم لقياس الثني الأمامي /
    الانبساط الخلفي من لقطة جانبية."""
    ear = np.mean([landmarks[LEFT_EAR][:2], landmarks[RIGHT_EAR][:2]], axis=0)
    shoulder = np.mean([landmarks[LEFT_SHOULDER][:2], landmarks[RIGHT_SHOULDER][:2]], axis=0)
    return np.array(ear) - np.array(shoulder)


def head_tilt_vector(landmarks):
    """متجه خط العينين/الأذنين في مستوى (x,y) - يُستخدم لقياس الميل
    الجانبي من لقطة أمامية."""
    left = np.array(landmarks[LEFT_EAR][:2])
    right = np.array(landmarks[RIGHT_EAR][:2])
    return right - left


def rotation_asymmetry(landmarks):
    """مؤشر الدوران: نسبة عمق (z) الأذنين وموقع الأنف أفقياً بالنسبة لمنتصف
    الكتفين. كلما زاد الدوران زاد الفارق بين عمق الأذن اليمنى واليسرى، وابتعد
    الأنف عن خط المنتصف."""
    left_ear_z = landmarks[LEFT_EAR][2]
    right_ear_z = landmarks[RIGHT_EAR][2]
    depth_diff = right_ear_z - left_ear_z  # موجب = دوران نحو اليسار تقريباً

    nose_x = landmarks[NOSE][0]
    shoulder_mid_x = np.mean([landmarks[LEFT_SHOULDER][0], landmarks[RIGHT_SHOULDER][0]])
    shoulder_width = abs(landmarks[LEFT_SHOULDER][0] - landmarks[RIGHT_SHOULDER][0]) or 1e-6
    lateral_offset = (nose_x - shoulder_mid_x) / shoulder_width

    return {"depth_diff": float(depth_diff), "lateral_offset": float(lateral_offset)}


VERTICAL = np.array([0.0, -1.0])   # لأعلى في إحداثيات الصورة (y ينمو للأسفل)
HORIZONTAL = np.array([1.0, 0.0])


def pitch_angle_deg(landmarks):
    return _angle_between(head_pitch_vector(landmarks), VERTICAL)


def tilt_angle_deg(landmarks):
    return _angle_between(head_tilt_vector(landmarks), HORIZONTAL)


def rom_from_neutral_and_peak(neutral_landmarks, peak_landmarks, direction: str) -> float:
    """يحسب مدى الحركة (بالدرجات) بمقارنة وضعية الذروة بالوضعية المحايدة،
    بنفس منطق تصفير الجهاز الرقمي الموصوف في الأطروحة."""
    if direction in ("flexion", "extension"):
        a0 = pitch_angle_deg(neutral_landmarks)
        a1 = pitch_angle_deg(peak_landmarks)
        return abs(a1 - a0)
    if direction in ("lateral_r", "lateral_l"):
        a0 = tilt_angle_deg(neutral_landmarks)
        a1 = tilt_angle_deg(peak_landmarks)
        return abs(a1 - a0)
    if direction in ("rotation_r", "rotation_l"):
        r0 = rotation_asymmetry(neutral_landmarks)
        r1 = rotation_asymmetry(peak_landmarks)
        # تقريب: نحول فرق الإزاحة الأفقية النسبية الى درجة تقريبية
        # (معايرة تجريبية بسيطة: إزاحة كاملة لعرض الكتف ~ 70 درجة دوران)
        delta = abs(r1["lateral_offset"] - r0["lateral_offset"])
        return float(delta * 70.0)
    raise ValueError(f"اتجاه غير معروف: {direction}")


# ------------------------------------------------------------------ وضعية الرأس من الوجه (أدق للأمامية)
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def matrix_to_head_angles(M):
    """مصفوفة تحويل الوجه 4x4 (MediaPipe: x يمينًا، y للأعلى، z نحو الكاميرا) -> (pitch, yaw, roll) بالدرجات.
    الاتفاقية نفسها المستخدمة في التطبيق على الصورة الخام (غير المعكوسة):
      pitch + = ثني أمامي (النظر للأسفل)،  yaw − = دوران نحو يمين الشخص،  roll − = ميل نحو يمين الشخص."""
    import math
    R = np.asarray(M, float)[:3, :3]
    n = R @ np.array([0.0, 0.0, 1.0])   # اتجاه الأنف
    u = R @ np.array([0.0, 1.0, 0.0])   # اتجاه أعلى الرأس
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -n[1]))))
    yaw = math.degrees(math.atan2(n[0], n[2]))
    roll = math.degrees(math.atan2(u[0], u[1]))
    return pitch, yaw, roll


class HeadPoseEstimator:
    """MediaPipe FaceLandmarker: يعطي زوايا الرأس الثلاث مباشرة من الوجه (جيد حتى ~60° دوران)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(FACE_MODEL_PATH):
            import urllib.request
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
        self._mp = mp
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)

    def estimate(self, image_rgb):
        """يعيد dict(pitch,yaw,roll) أو None إن لم يُكتشف وجه."""
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb))
        if not res.facial_transformation_matrixes:
            return None
        p, y, r = matrix_to_head_angles(res.facial_transformation_matrixes[0])
        return {"pitch": p, "yaw": y, "roll": r}
