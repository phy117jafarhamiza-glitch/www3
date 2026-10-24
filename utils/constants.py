# -*- coding: utf-8 -*-
"""
ثوابت التطبيق — النسخة المعدّلة (AI-NeckRehab) المتوافقة مع المقترح المعدّل:
ثلاث مجموعات، أربع نقاط قياس (T0–T3)، ومتغيرات أساسية وثانوية محددة.
"""

# المجموعات الثلاث (الجدول 1 في المقترح)
GROUP_AI = "G1: تجريبية (AI كامل)"
GROUP_FIXED = "G2: ضابطة مقارِنة (جرعة ثابتة)"
GROUP_CONV = "G3: ضابطة تقليدية"
GROUPS = [GROUP_AI, GROUP_FIXED, GROUP_CONV]

# نقاط القياس
STAGES = [
    "T0 قبلي",
    "T1 أسبوع 4",
    "T2 بعدي (أسبوع 8)",
    "T3 متابعة (أسبوع 12)",
]
SESSION_STAGES = STAGES  # توافق مع الاسم القديم

DIRECTIONS = ["flex", "ext", "lat_r", "lat_l", "rot_r", "rot_l"]
DIRECTION_AR = {
    "flex": "ثني أمامي", "ext": "بسط خلفي",
    "lat_r": "ميل أيمن", "lat_l": "ميل أيسر",
    "rot_r": "دوران أيمن", "rot_l": "دوران أيسر",
}
NECK_DIRECTIONS = [DIRECTION_AR[d] for d in DIRECTIONS]

# متغيرات القياس: المفتاح -> (الاسم، الوحدة، الحد الأدنى، الحد الأعلى، النوع)
TEST_META = {
    "nprs_move": ("الألم أثناء الحركة (NPRS) — أساسي", "0–10", 0.0, 10.0, "primary"),
    "nprs_rest": ("الألم في الراحة (NPRS)", "0–10", 0.0, 10.0, "secondary"),
}
for _d in DIRECTIONS:
    TEST_META[f"rom_{_d}"] = (f"مدى الحركة — {DIRECTION_AR[_d]}", "درجة", 0.0, 120.0, "secondary")
for _d in DIRECTIONS:
    TEST_META[f"str_{_d}"] = (f"القوة الإيزومترية — {DIRECTION_AR[_d]}", "كغم", 0.0, 80.0, "secondary")
TEST_META.update({
    "emg_scm": ("EMG — القصية الترقوية الخشائية (% MVC أثناء CCFT)", "% MVC", 0.0, 300.0, "secondary"),
    "emg_scalene": ("EMG — الشقرافية الأمامية (% MVC أثناء CCFT)", "% MVC", 0.0, 300.0, "secondary"),
    "emg_trap": ("EMG — شبه المنحرفة العلوية (% MVC أثناء CCFT)", "% MVC", 0.0, 300.0, "secondary"),
    "ccft_level": ("CCFT — أعلى مستوى ضغط مثبَّت بجودة", "mmHg", 20.0, 30.0, "secondary"),
    "jpe": ("دقة إعادة وضع الرأس (JPE)", "درجة", 0.0, 30.0, "secondary"),
    "tsk11": ("الخوف الحركي (TSK-11)", "11–44", 11.0, 44.0, "secondary"),
})
TEST_META.update({
    # عتبة ألم الضغط (PPT) — مقياس ألم موضوعي نسبيًا
    "ppt_trap_r": ("PPT — شبه المنحرفة العلوية يمين", "kPa", 0.0, 1000.0, "secondary"),
    "ppt_trap_l": ("PPT — شبه المنحرفة العلوية يسار", "kPa", 0.0, 1000.0, "secondary"),
    "ppt_scm_r": ("PPT — القصية الترقوية الخشائية يمين", "kPa", 0.0, 1000.0, "secondary"),
    "ppt_scm_l": ("PPT — القصية الترقوية الخشائية يسار", "kPa", 0.0, 1000.0, "secondary"),
    # التحمل
    "end_flex": ("تحمل المثنيات (ثبات)", "ث", 0.0, 400.0, "secondary"),
    "end_ext": ("تحمل الباسطات (ثبات)", "ث", 0.0, 600.0, "secondary"),
    # التحكم الحركي
    "mc_track_rmse": ("دقة التتبع بالرأس (RMSE)", "درجة", 0.0, 40.0, "secondary"),
    "mc_sparc": ("سلاسة الحركة (SPARC)", "—", -15.0, 0.0, "secondary"),
    "mc_coupled": ("الحركة المرافقة غير المقصودة", "%", 0.0, 100.0, "secondary"),
    "mc_vel_asym": ("عدم تماثل السرعة يمين/يسار", "%", 0.0, 100.0, "secondary"),
    # مؤشرات مركّبة (تُحسب في صفحتي التوازن والتحكم)
    "nmci_balance": ("مؤشر التوازن العصبي العضلي", "0–100", 0.0, 100.0, "secondary"),
    "nmci_motor": ("مؤشر التحكم الحركي", "0–100", 0.0, 100.0, "secondary"),
    "nmci_total": ("مؤشر التحكم العصبي العضلي الكلي (NMCI)", "0–100", 0.0, 100.0, "secondary"),
})
TEST_META.update({
    "kin_guard": ("مؤشر الحذر الحركي من الفيديو (تجريبي)", "0–100", 0.0, 100.0, "secondary"),
    "rt_ms": ("زمن الاستجابة الحركية للرأس", "ms", 0.0, 3000.0, "secondary"),
    "mt_ms": ("زمن الحركة حتى الذروة", "ms", 0.0, 3000.0, "secondary"),
    "rt_acc": ("دقة الاستجابة الاتجاهية", "%", 0.0, 100.0, "secondary"),
})
TEST_NAMES = {k: f"{v[0]} [{v[1]}]" for k, v in TEST_META.items()}
NUMERIC_TEST_KEYS = list(TEST_META.keys())
PRIMARY_KEY = "nprs_move"

# متغيرات قبلية للتكافؤ (تُخزَّن في بيانات اللاعب)
COVARIATES = {
    "age": "العمر (سنة)",
    "bmi": "مؤشر كتلة الجسم",
    "injury_days": "مدة الإصابة (يوم)",
    "training_years": "سنوات التدريب",
}

# عضلات EMG السطحي (لا يقيس EMG السطحي المثنيات العميقة مباشرة — تُقيَّم بـ CCFT)
EMG_MUSCLES = [
    "القصية الترقوية الخشائية (SCM)",
    "الشقرافية الأمامية (Anterior Scalene)",
    "شبه المنحرفة العلوية (Upper Trapezius)",
    "الرافعة للكتف (Levator Scapulae)",
]

# فئات EMG-Net
EMG_QUALITY_LABELS = ["ضعيف/غير متوازن", "طبيعي", "مفرط/تعويضي"]

RESISTANCE_LEVELS = ["منخفضة", "متوسطة", "عالية"]

# مراحل البرنامج
def phase_of_week(week: int) -> int:
    return 1 if week <= 2 else (2 if week <= 5 else 3)

PHASE_NAMES = {1: "المرحلة 1: تهدئة وتنشيط", 2: "المرحلة 2: تقوية", 3: "المرحلة 3: تحكم عصبي عضلي ونوعي"}

# سقوف المقاومة الأسبوعية (% من MVC الأساسية) — مقترحة وتُعتمد مع المشرف
WEEK_RESISTANCE_CAP = {1: 30, 2: 30, 3: 45, 4: 45, 5: 60, 6: 60, 7: 70, 8: 70}

# جدول الجرعة الثابتة لـ G2 (من الصيغة الأصلية للمقترح)
FIXED_DOSE = {
    1: dict(reps=8, sets=2, rest_sets=60),  2: dict(reps=10, sets=2, rest_sets=45),
    3: dict(reps=12, sets=3, rest_sets=45), 4: dict(reps=12, sets=3, rest_sets=45),
    5: dict(reps=12, sets=3, rest_sets=30), 6: dict(reps=14, sets=3, rest_sets=30),
    7: dict(reps=15, sets=4, rest_sets=30), 8: dict(reps=15, sets=4, rest_sets=30),
}
SESSION_MINUTES = {1: 15, 2: 20, 3: 25, 4: 30, 5: 32, 6: 35, 7: 38, 8: 40}

DB_PATH = "data/neck_rehab_v2.db"
MODEL_DIR = "data/models"


# ----------------------------------------------------------------------
# البرنامج التأهيلي: 8 أسابيع × 3 جلسات (جدول 1 و2 في ملف البرنامج)
LEVELS = {1: "1 — سهل", 2: "2 — متوسط", 3: "3 — صعب"}


def level_of_week(week: int) -> int:
    return 1 if week <= 2 else (2 if week <= 5 else 3)


# (إحماء، رئيسي، ختام) بالدقائق — المجموع = SESSION_MINUTES
SESSION_SPLIT = {1: (3, 9, 3), 2: (4, 12, 4), 3: (5, 15, 5), 4: (5, 20, 5),
                 5: (5, 22, 5), 6: (5, 25, 5), 7: (5, 28, 5), 8: (5, 30, 5)}
SESSIONS_PER_WEEK = 3

# بنك التمارين المقترح (17) — يخضع لتحكيم الخبراء (Lawshe CVR)
# tracking: cam=كاميرا، emg=EMG، force=قوة، press=مقياس ضغط CCFT، imu=IMU، coach=ملاحظة المدرب
EXERCISES = [
    dict(id=1, name="تراجع الذقن (Chin Tuck) مستلقيًا", aim="تنشيط المثنيات العميقة", levels=[1], weeks="1–2", dose="2 × 8، ثبات 5–10 ث", tracking="press+emg"),
    dict(id=2, name="انقباض إيزومتري خفيف باليد (ثني، بسط، ميل)", aim="تنشيط عام دون حركة", levels=[1], weeks="1–2", dose="2 × 8، ثبات 5 ث", tracking="force+emg"),
    dict(id=3, name="حركة رقبة نشطة بطيئة في 6 اتجاهات", aim="استعادة المدى وتقليل التيبس", levels=[1, 2], weeks="1–4", dose="2 × 8–10", tracking="cam", cam=dict(mode="rep", axis="multi", target=40, safe=70, min_rep_s=3.0)),
    dict(id=4, name="شد الكتفين للخلف (Scapular Retraction)", aim="تحكم لوحي كتفي", levels=[1], weeks="1–2", dose="2 × 10، ثبات 3 ث", tracking="cam"),
    dict(id=5, name="إطالة المعينة العليا والرافعة للكتف", aim="تقليل التوتر وتهدئة الألم", levels=[1, 2, 3], weeks="1–8", dose="20–30 ث لكل جانب", tracking="cam", cam=dict(mode="hold", axis="roll", target=25, safe=45, hold_s=20)),
    dict(id=6, name="تدريب CCFT بمقياس الضغط (22 → 26 mmHg)", aim="تحمل وتحكم المثنيات العميقة", levels=[2], weeks="3–5", dose="3 × 10 ث لكل مستوى ضغط", tracking="press+emg"),
    dict(id=7, name="انقباض إيزومتري بالمطاط أو الجهاز (ثني، بسط، ميل)", aim="تقوية تدريجية", levels=[2], weeks="3–5", dose="3 × 12، ثبات 5 ث", tracking="force+emg"),
    dict(id=8, name="رفع الرأس من الانبطاح", aim="تقوية الباسطات", levels=[2], weeks="3–5", dose="3 × 10–12", tracking="cam (جانبية)", cam=dict(mode="rep", axis="pitch", sign=-1, target=25, safe=40, min_rep_s=3.0)),
    dict(id=9, name="رفع الرأس جانبيًا من الاستلقاء الجانبي", aim="تقوية الماثلات", levels=[2], weeks="3–5", dose="3 × 10–12", tracking="cam (أمامية)", cam=dict(mode="rep", axis="roll", target=25, safe=40, min_rep_s=3.0)),
    dict(id=10, name="دوران الرقبة بمقاومة المطاط", aim="تقوية المدورات", levels=[2], weeks="4–5", dose="3 × 12", tracking="cam+force", cam=dict(mode="rep", axis="yaw", target=45, safe=70, min_rep_s=3.0)),
    dict(id=11, name="تمرين الشد بالمطاط جالسًا (Row)", aim="تقوية مثبتات الكتف", levels=[2], weeks="3–5", dose="3 × 12", tracking="cam"),
    dict(id=12, name="مقاومة ديناميكية في 6 اتجاهات بالجهاز", aim="تقوية وظيفية", levels=[3], weeks="6–8", dose="3–4 × 14–15", tracking="جهاز+emg+cam", cam=dict(mode="rep", axis="multi", target=45, safe=75, min_rep_s=3.0)),
    dict(id=13, name="تثبيت الرأس عند ضغط مفاجئ (Perturbation)", aim="تحكم عصبي عضلي وتثبيت", levels=[3], weeks="6–8", dose="3 × 30 ث", tracking="emg+imu"),
    dict(id=14, name="تتبع هدف بالرأس (JPE)", aim="دقة الحس العميق والتحكم الحركي", levels=[3], weeks="6–8", dose="3 × 1 د", tracking="cam+imu"),
    dict(id=15, name="ثبات الرأس في البلانك أو وضع الأربع نقاط", aim="تثبيت الرقبة مع الجذع والكتف", levels=[3], weeks="6–8", dose="3 × 20–30 ث", tracking="cam (جانبية)", cam=dict(mode="neutral_hold", axis="pitch", target=8, safe=15, hold_s=25)),
    dict(id=16, name="ثني وبسط بطيء متحكم (انقباض لا مركزي)", aim="تحمل وتحكم أثناء الإطالة العضلية", levels=[3], weeks="7–8", dose="3 × 8–10، 4 ث للنزول", tracking="cam+force", cam=dict(mode="rep", axis="pitch", sign=1, target=35, safe=55, min_rep_s=4.0)),
    dict(id=17, name="محاكاة مسكات الفنون القتالية بتحكم المدرب", aim="نقل الأداء إلى الرياضة", levels=[3], weeks="7–8", dose="3 × 20–30 ث", tracking="coach+emg"),
]
EXERCISE_BY_ID = {e["id"]: e for e in EXERCISES}


def exercises_for_week(week: int):
    lv = level_of_week(week)
    return [e for e in EXERCISES if lv in e["levels"]]
