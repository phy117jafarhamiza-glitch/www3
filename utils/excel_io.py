# -*- coding: utf-8 -*-
"""استيراد/تصدير Excel للاعبين والقياسات (قالب جاهز + تحقق من البيانات)."""
import io
import re

import pandas as pd

from db import add_subject, list_subjects, upsert_measurement
from utils.constants import GROUPS, STAGES, TEST_META, TEST_NAMES

SUBJECT_COLS = {
    "code": "رمز اللاعب", "name": "الاسم", "group": "المجموعة", "age": "العمر", "sport": "الرياضة",
    "height_cm": "الطول (سم)", "weight_kg": "الوزن (كغم)", "injury_days": "مدة الإصابة (يوم)",
    "training_years": "سنوات التدريب", "injury_note": "ملاحظات الإصابة",
}
STAGE_COL = "نقطة القياس"
# المتغيرات المعروضة في القالب (الاختبارات الستة)
TEMPLATE_KEYS = ([f"str_{d}" for d in ("flex", "ext", "lat_r", "lat_l", "rot_r", "rot_l")]
                 + [f"rom_{d}" for d in ("flex", "ext", "lat_r", "lat_l", "rot_r", "rot_l")]
                 + ["nprs_move", "nprs_rest", "tsk11", "kin_guard", "emg_scm", "emg_scalene", "emg_trap",
                    "rt_ms", "mt_ms", "rt_acc"])


def _norm(s):
    return re.sub(r"[\s_\-—–:()\[\]/\\]+", "", str(s)).lower()


def _header_map():
    m = {}
    for k, lab in SUBJECT_COLS.items():
        m[_norm(k)] = ("subj", k)
        m[_norm(lab)] = ("subj", k)
    m[_norm(STAGE_COL)] = ("stage", None)
    m[_norm("stage")] = ("stage", None)
    for k in TEST_META:
        m[_norm(k)] = ("meas", k)
        m[_norm(TEST_NAMES[k])] = ("meas", k)
        m[_norm(TEST_META[k][0])] = ("meas", k)
    return m


def template_bytes() -> bytes:
    """قالب Excel: ورقة «القياسات» (صف لكل لاعب × نقطة قياس) وورقة «تعليمات»."""
    cols = list(SUBJECT_COLS.values()) + [STAGE_COL] + [TEST_NAMES[k] for k in TEMPLATE_KEYS]
    ex = pd.DataFrame([{c: None for c in cols} for _ in range(2)])
    ex.loc[0, SUBJECT_COLS["code"]] = "P001"
    ex.loc[0, SUBJECT_COLS["name"]] = "اسم اللاعب"
    ex.loc[0, SUBJECT_COLS["group"]] = GROUPS[0]
    ex.loc[0, SUBJECT_COLS["age"]] = 22
    ex.loc[0, STAGE_COL] = STAGES[0]
    ex.loc[0, TEST_NAMES["nprs_move"]] = 5
    ex.loc[1, SUBJECT_COLS["code"]] = "P001"
    ex.loc[1, STAGE_COL] = STAGES[1]
    ex.loc[1, TEST_NAMES["nprs_move"]] = 3
    info = pd.DataFrame({"تعليمات": [
        "صف واحد لكل لاعب في كل نقطة قياس (يمكن تكرار اللاعب في أكثر من صف).",
        "يُعرَّف اللاعب برمزه أو باسمه؛ إن لم يكن موجودًا يُنشأ تلقائيًا (يلزم الاسم والمجموعة).",
        "المجموعة: " + " / ".join(GROUPS) + " (تُقبل أيضًا G1 وG2 وG3).",
        "نقطة القياس: " + " / ".join(STAGES) + " (تُقبل أيضًا T0 وT1 وT2 وT3). الافتراضي T0.",
        "اترك الخلية فارغة إذا لم يُقَس المتغير. الألم من 0 إلى 10.",
        "يمكن استخدام أسماء الأعمدة العربية في القالب أو مفاتيحها الإنجليزية (مثل str_flex وrom_flex).",
    ]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        ex.to_excel(xw, sheet_name="القياسات", index=False)
        info.to_excel(xw, sheet_name="تعليمات", index=False)
    return buf.getvalue()


def _group(v):
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    for g in GROUPS:
        if s == g or s.upper().startswith(g[:2]):
            return g
    if "تجريب" in s:
        return GROUPS[0]
    if "ثابت" in s or "مقارن" in s:
        return GROUPS[1]
    if "تقليد" in s or "ضابط" in s:
        return GROUPS[2]
    return None


def _stage(v):
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return STAGES[0]
    for st in STAGES:
        if s == st or s.upper().startswith(st[:2]):
            return st
    return None


def read_any(file_or_bytes, sheet=None) -> pd.DataFrame:
    name = getattr(file_or_bytes, "name", "")
    data = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    if str(name).lower().endswith((".csv", ".txt")):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)


def parse(df: pd.DataFrame):
    """يعيد (rows, problems): rows = قائمة dict(subject=..., stage=..., values={key: float}); problems = رسائل."""
    hm = _header_map()
    colmap, unknown = {}, []
    for c in df.columns:
        t = hm.get(_norm(c))
        if t:
            colmap[c] = t
        else:
            unknown.append(str(c))
    problems = []
    if unknown:
        problems.append("أعمدة غير معروفة تم تجاهلها: " + "، ".join(unknown[:12]))
    if not any(t == ("subj", "code") or t == ("subj", "name") for t in colmap.values()):
        return [], problems + ["يلزم عمود «رمز اللاعب» أو «الاسم»."]
    rows = []
    for i, r in df.iterrows():
        subj, stage, vals = {}, STAGES[0], {}
        for c, (kind, key) in colmap.items():
            v = r[c]
            if pd.isna(v):
                continue
            if kind == "subj":
                subj[key] = v
            elif kind == "stage":
                st = _stage(v)
                if st is None:
                    problems.append(f"السطر {i + 2}: نقطة قياس غير معروفة «{v}».")
                    stage = None
                else:
                    stage = st
            else:
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    problems.append(f"السطر {i + 2}: قيمة غير رقمية في {key}: «{v}».")
                    continue
                lo, hi = TEST_META[key][2], TEST_META[key][3]
                if not (lo <= x <= hi):
                    problems.append(f"السطر {i + 2}: {key}={x:g} خارج المدى [{lo:g}, {hi:g}] وتم تجاهله.")
                    continue
                vals[key] = x
        if not subj.get("code") and not subj.get("name"):
            continue
        if stage is None:
            continue
        if "group" in subj:
            g = _group(subj["group"])
            if g is None:
                problems.append(f"السطر {i + 2}: مجموعة غير معروفة «{subj['group']}».")
            subj["group"] = g
        rows.append({"subject": subj, "stage": stage, "values": vals, "line": i + 2})
    return rows, problems


def import_rows(rows, create_missing=True):
    """يكتب الصفوف في قاعدة البيانات. يعيد ملخّصًا وقائمة مشكلات."""
    subjects = list_subjects()
    by_code = {str(s["code"]).strip(): s["id"] for s in subjects if s.get("code")}
    by_name = {str(s["name"]).strip(): s["id"] for s in subjects}
    created = measured = skipped = 0
    problems = []
    for r in rows:
        sj = r["subject"]
        code = str(sj.get("code", "")).strip() or None
        name = str(sj.get("name", "")).strip() or None
        sid = by_code.get(code) if code else None
        if sid is None and name:
            sid = by_name.get(name)
        if sid is None:
            if not create_missing or not name or not sj.get("group"):
                problems.append(f"السطر {r['line']}: لاعب غير موجود ولا تكفي بياناته للإنشاء (الاسم والمجموعة).")
                skipped += 1
                continue
            def num(k, cast=float):
                try:
                    return cast(sj[k]) if k in sj else None
                except (TypeError, ValueError):
                    return None
            sid = add_subject(name, sj["group"], num("age", int), str(sj.get("sport", "") or ""),
                              str(sj.get("injury_note", "") or ""), num("height_cm"), num("weight_kg"),
                              num("injury_days", int), num("training_years"), code)
            if not sid:
                sid = list_subjects()[-1]["id"]
            if code:
                by_code[code] = sid
            if name:
                by_name[name] = sid
            created += 1
        for k, v in r["values"].items():
            upsert_measurement(sid, r["stage"], k, v)
            measured += 1
    return {"لاعبون جدد": created, "قياسات مكتوبة": measured, "صفوف متجاهلة": skipped}, problems
