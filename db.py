# -*- coding: utf-8 -*-
"""
طبقة قاعدة البيانات (SQLite) — النسخة المعدّلة AI-NeckRehab.

تحذير مهم للنشر على Streamlit Community Cloud: نظام الملفات هناك مؤقت،
وقد تُفقد قاعدة البيانات عند إعادة التشغيل. استخدم صفحة «النسخ الاحتياطي»
لتنزيل نسخة بعد كل جلسة، أو اربط التطبيق بقاعدة بيانات خارجية.
"""
import io
import os
import shutil
import sqlite3
from contextlib import contextmanager

import pandas as pd

from utils.constants import DB_PATH


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                name TEXT NOT NULL,
                group_name TEXT NOT NULL,
                age INTEGER, sport TEXT,
                height_cm REAL, weight_kg REAL, bmi REAL,
                injury_days INTEGER, training_years REAL,
                injury_note TEXT,
                baseline_mvc REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL, stage TEXT NOT NULL, test_key TEXT NOT NULL,
                value REAL, notes TEXT, recorded_at TEXT DEFAULT (datetime('now')),
                UNIQUE(subject_id, stage, test_key),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS emg_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, stage TEXT, muscle TEXT, source_file TEXT,
                predicted_label TEXT, confidence REAL,
                rms REAL, mav REAL, zero_crossings REAL, waveform_length REAL,
                model_backend TEXT, created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS ai_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, week INTEGER, session_no INTEGER, set_no INTEGER,
                direction TEXT, emg_pct_mvc REAL, quality_normal REAL, quality_label TEXT,
                head_angle_pct REAL, velocity REAL, nprs_now REAL, nprs_session_start REAL,
                prev_resistance REAL,
                rec_resistance REAL, rec_reps INTEGER, rec_sets INTEGER,
                safety_action TEXT, safety_messages TEXT,
                final_resistance REAL, created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS adverse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, week INTEGER, description TEXT, severity TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS pose_rom_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, stage TEXT, source TEXT, direction TEXT, value REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS rom_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, direction TEXT, camera_value REAL, reference_value REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS pain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, week INTEGER, session_no INTEGER, exercise_id INTEGER, set_no INTEGER,
                nprs REAL, angle_deg REAL, emg_pct_mvc REAL, note TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS session_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, week INTEGER, session_no INTEGER, minutes REAL, completed INTEGER,
                rpe REAL, note TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(subject_id, week, session_no),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS coach_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, week INTEGER, session_no INTEGER, exercise_id INTEGER,
                reps INTEGER, mean_rom REAL, max_velocity REAL, too_fast INTEGER, short_rom INTEGER,
                compensation INTEGER, over_limit INTEGER, quality_score REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL, based_on_stage TEXT, program_json TEXT, note TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS reaction_trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER, stage TEXT, trial_no INTEGER, direction TEXT,
                rt_ms REAL, mt_ms REAL, peak_velocity REAL, correct INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS dose_expert_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT, quality_normal REAL, emg_pct_mvc REAL, head_angle_pct REAL,
                velocity REAL, nprs_now REAL, week INTEGER, prev_resistance REAL,
                exp_resistance REAL, exp_reps INTEGER, exp_sets INTEGER, rater TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )


# ---------------------------------------------------------------- subjects
def add_subject(name, group_name, age=None, sport=None, injury_note=None, height_cm=None,
                weight_kg=None, injury_days=None, training_years=None, code=None):
    bmi = None
    if height_cm and weight_kg and height_cm > 0:
        bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO subjects (code,name,group_name,age,sport,height_cm,weight_kg,bmi,
               injury_days,training_years,injury_note) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (code, name, group_name, age, sport, height_cm, weight_kg, bmi,
             injury_days, training_years, injury_note))
        return cur.lastrowid


def list_subjects(group_name=None):
    with get_conn() as conn:
        if group_name:
            rows = conn.execute("SELECT * FROM subjects WHERE group_name=? ORDER BY id", (group_name,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM subjects ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def update_baseline_mvc(subject_id, mvc):
    with get_conn() as conn:
        conn.execute("UPDATE subjects SET baseline_mvc=? WHERE id=?", (mvc, subject_id))


def delete_subject(subject_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))


# ------------------------------------------------------------ measurements
def upsert_measurement(subject_id, stage, test_key, value, notes=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO measurements (subject_id,stage,test_key,value,notes) VALUES (?,?,?,?,?)
               ON CONFLICT(subject_id,stage,test_key) DO UPDATE SET value=excluded.value,
               notes=excluded.notes, recorded_at=datetime('now')""",
            (subject_id, stage, test_key, value, notes))


def delete_measurement(subject_id, stage, test_key):
    with get_conn() as conn:
        conn.execute("DELETE FROM measurements WHERE subject_id=? AND stage=? AND test_key=?",
                     (subject_id, stage, test_key))


def get_measurements(subject_id=None, stage=None, test_key=None):
    q = ("SELECT m.*, s.name AS subject_name, s.group_name FROM measurements m "
         "JOIN subjects s ON s.id=m.subject_id WHERE 1=1")
    params = []
    for col, val in (("m.subject_id", subject_id), ("m.stage", stage), ("m.test_key", test_key)):
        if val is not None:
            q += f" AND {col}=?"
            params.append(val)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


# ------------------------------------------------------------- emg analyses
def add_emg_analysis(subject_id, stage, muscle, source_file, predicted_label, confidence, features, model_backend):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO emg_analyses (subject_id,stage,muscle,source_file,predicted_label,confidence,
               rms,mav,zero_crossings,waveform_length,model_backend) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (subject_id, stage, muscle, source_file, predicted_label, confidence,
             features.get("rms"), features.get("mav"), features.get("zc"), features.get("wl"), model_backend))


def list_emg_analyses(subject_id=None):
    q = "SELECT e.*, s.name AS subject_name FROM emg_analyses e LEFT JOIN subjects s ON s.id=e.subject_id"
    params = []
    if subject_id:
        q += " WHERE e.subject_id=?"
        params.append(subject_id)
    q += " ORDER BY e.id DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


# --------------------------------------------------------------- AI sets log
def add_ai_set(**kw):
    cols = ["subject_id", "week", "session_no", "set_no", "direction", "emg_pct_mvc", "quality_normal",
            "quality_label", "head_angle_pct", "velocity", "nprs_now", "nprs_session_start",
            "prev_resistance", "rec_resistance", "rec_reps", "rec_sets", "safety_action",
            "safety_messages", "final_resistance"]
    with get_conn() as conn:
        conn.execute(f"INSERT INTO ai_sets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [kw.get(c) for c in cols])


def list_ai_sets(subject_id=None, limit=500):
    q = "SELECT a.*, s.name AS subject_name FROM ai_sets a LEFT JOIN subjects s ON s.id=a.subject_id"
    params = []
    if subject_id:
        q += " WHERE a.subject_id=?"
        params.append(subject_id)
    q += " ORDER BY a.id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


def last_ai_set(subject_id):
    rows = list_ai_sets(subject_id, 1)
    return rows[0] if rows else None


# ------------------------------------------------------------- adverse events
def add_adverse_event(subject_id, week, description, severity):
    with get_conn() as conn:
        conn.execute("INSERT INTO adverse_events (subject_id,week,description,severity) VALUES (?,?,?,?)",
                     (subject_id, week, description, severity))


def list_adverse_events(subject_id=None):
    q = "SELECT e.*, s.name AS subject_name FROM adverse_events e LEFT JOIN subjects s ON s.id=e.subject_id"
    params = []
    if subject_id:
        q += " WHERE e.subject_id=?"
        params.append(subject_id)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q + " ORDER BY e.id DESC", params).fetchall()]


# ------------------------------------------------------------- pose / validation
def add_pose_rom(subject_id, stage, source, direction, value):
    with get_conn() as conn:
        conn.execute("INSERT INTO pose_rom_sessions (subject_id,stage,source,direction,value) VALUES (?,?,?,?,?)",
                     (subject_id, stage, source, direction, value))


def list_pose_rom(subject_id=None):
    q = "SELECT p.*, s.name AS subject_name FROM pose_rom_sessions p LEFT JOIN subjects s ON s.id=p.subject_id"
    params = []
    if subject_id:
        q += " WHERE p.subject_id=?"
        params.append(subject_id)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q + " ORDER BY p.id DESC", params).fetchall()]


def add_rom_validation(subject_id, direction, camera_value, reference_value):
    with get_conn() as conn:
        conn.execute("INSERT INTO rom_validation (subject_id,direction,camera_value,reference_value) VALUES (?,?,?,?)",
                     (subject_id, direction, camera_value, reference_value))


def list_rom_validation():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM rom_validation ORDER BY id").fetchall()]


# ------------------------------------------------------------ expert dose labels
def add_dose_label(row: dict):
    cols = ["direction", "quality_normal", "emg_pct_mvc", "head_angle_pct", "velocity", "nprs_now", "week",
            "prev_resistance", "exp_resistance", "exp_reps", "exp_sets", "rater"]
    with get_conn() as conn:
        conn.execute(f"INSERT INTO dose_expert_labels ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [row.get(c) for c in cols])


def list_dose_labels():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM dose_expert_labels ORDER BY id").fetchall()]


# ------------------------------------------------------- pain / sessions / coach
def _rows(q, params=()):
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


def add_pain(subject_id, week, session_no, exercise_id, set_no, nprs, angle_deg=None, emg_pct_mvc=None, note=None):
    with get_conn() as conn:
        conn.execute("INSERT INTO pain_log (subject_id,week,session_no,exercise_id,set_no,nprs,angle_deg,emg_pct_mvc,note) "
                     "VALUES (?,?,?,?,?,?,?,?,?)",
                     (subject_id, week, session_no, exercise_id, set_no, nprs, angle_deg, emg_pct_mvc, note))


def list_pain(subject_id=None):
    if subject_id:
        return _rows("SELECT * FROM pain_log WHERE subject_id=? ORDER BY id", (subject_id,))
    return _rows("SELECT * FROM pain_log ORDER BY id")


def upsert_session(subject_id, week, session_no, minutes, completed, rpe=None, note=None):
    with get_conn() as conn:
        conn.execute("""INSERT INTO session_log (subject_id,week,session_no,minutes,completed,rpe,note)
                        VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(subject_id,week,session_no) DO UPDATE SET
                        minutes=excluded.minutes, completed=excluded.completed, rpe=excluded.rpe, note=excluded.note""",
                     (subject_id, week, session_no, minutes, int(bool(completed)), rpe, note))


def list_sessions(subject_id=None):
    if subject_id:
        return _rows("SELECT * FROM session_log WHERE subject_id=? ORDER BY week, session_no", (subject_id,))
    return _rows("SELECT * FROM session_log ORDER BY subject_id, week, session_no")


def add_coach_session(**kw):
    cols = ["subject_id", "week", "session_no", "exercise_id", "reps", "mean_rom", "max_velocity", "too_fast",
            "short_rom", "compensation", "over_limit", "quality_score"]
    with get_conn() as conn:
        conn.execute(f"INSERT INTO coach_sessions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [kw.get(c) for c in cols])


def list_coach_sessions(subject_id=None):
    if subject_id:
        return _rows("SELECT * FROM coach_sessions WHERE subject_id=? ORDER BY id DESC", (subject_id,))
    return _rows("SELECT * FROM coach_sessions ORDER BY id DESC")


# ------------------------------------------------------- programs / reaction
def save_program(subject_id, program: dict, based_on_stage=None, note=None):
    import json
    with get_conn() as conn:
        conn.execute("INSERT INTO programs (subject_id,based_on_stage,program_json,note) VALUES (?,?,?,?)",
                     (subject_id, based_on_stage, json.dumps(program, ensure_ascii=False), note))


def latest_program(subject_id):
    import json
    rows = _rows("SELECT * FROM programs WHERE subject_id=? ORDER BY id DESC LIMIT 1", (subject_id,))
    if not rows:
        return None
    r = rows[0]
    r["program"] = json.loads(r["program_json"])
    return r


def list_programs(subject_id=None):
    if subject_id:
        return _rows("SELECT id,subject_id,based_on_stage,note,created_at FROM programs WHERE subject_id=? ORDER BY id DESC", (subject_id,))
    return _rows("SELECT id,subject_id,based_on_stage,note,created_at FROM programs ORDER BY id DESC")


def add_reaction_trials(subject_id, stage, trials):
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO reaction_trials (subject_id,stage,trial_no,direction,rt_ms,mt_ms,peak_velocity,correct) VALUES (?,?,?,?,?,?,?,?)",
            [(subject_id, stage, t.get("trial_no"), t.get("direction"), t.get("rt_ms"), t.get("mt_ms"),
              t.get("peak_velocity"), int(bool(t.get("correct")))) for t in trials])


def list_reaction_trials(subject_id=None):
    if subject_id:
        return _rows("SELECT * FROM reaction_trials WHERE subject_id=? ORDER BY id DESC", (subject_id,))
    return _rows("SELECT * FROM reaction_trials ORDER BY id DESC")


# ---------------------------------------------------------------- backup/restore
EXPORT_TABLES = ["subjects", "measurements", "emg_analyses", "ai_sets", "adverse_events",
                 "pose_rom_sessions", "rom_validation", "dose_expert_labels", "pain_log", "session_log",
                 "coach_sessions", "programs", "reaction_trials"]


def db_bytes() -> bytes:
    init_db()
    with open(DB_PATH, "rb") as f:
        return f.read()


def restore_db(data: bytes):
    _ensure_dir()
    if os.path.exists(DB_PATH):
        shutil.copy(DB_PATH, DB_PATH + ".bak")
    with open(DB_PATH, "wb") as f:
        f.write(data)
    init_db()


def export_excel_bytes() -> bytes:
    init_db()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        with get_conn() as conn:
            for t in EXPORT_TABLES:
                pd.read_sql_query(f"SELECT * FROM {t}", conn).to_excel(xw, sheet_name=t, index=False)
            wide = pd.read_sql_query(
                "SELECT s.id AS subject_id, COALESCE(s.code, 'S' || s.id) AS code, s.group_name, m.stage, m.test_key, m.value "
                "FROM measurements m JOIN subjects s ON s.id=m.subject_id", conn)
        if len(wide):
            wide.pivot_table(index=["subject_id", "code", "group_name", "stage"], columns="test_key",
                             values="value", aggfunc="first").reset_index().to_excel(
                xw, sheet_name="wide_for_SPSS", index=False)
    return buf.getvalue()
