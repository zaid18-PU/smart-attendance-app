"""
SQLite persistence layer for Smart Attendance AI.
Keeps things dependency-light (no ORM) so it runs anywhere, including Colab.
"""
import sqlite3
import json
import time
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("ATTENDANCE_DB_PATH", "attendance.db")


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                vector TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                total_roster INTEGER NOT NULL,
                total_present INTEGER NOT NULL,
                total_absent INTEGER NOT NULL,
                summary TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id INTEGER,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# ---------- Students / Roster ----------

def add_student(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO students (name, created_at) VALUES (?, ?)",
            (name, time.time()),
        )
        conn.commit()
        return cur.lastrowid


def get_or_create_student(name: str) -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM students WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
    return add_student(name)


def add_embedding(student_id: int, vector: list):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO embeddings (student_id, vector, created_at) VALUES (?, ?, ?)",
            (student_id, json.dumps(vector), time.time()),
        )
        conn.commit()


def get_roster():
    """Returns list of {id, name, num_embeddings}"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id, s.name, COUNT(e.id) as n
            FROM students s
            LEFT JOIN embeddings e ON e.student_id = s.id
            GROUP BY s.id
            ORDER BY s.name
        """).fetchall()
    return [{"id": r[0], "name": r[1], "num_embeddings": r[2]} for r in rows]


def get_all_embeddings():
    """Returns dict: student_id -> {name, vectors: [list of float lists]}"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id, s.name, e.vector
            FROM students s
            JOIN embeddings e ON e.student_id = s.id
        """).fetchall()
    result = {}
    for sid, name, vec_json in rows:
        if sid not in result:
            result[sid] = {"name": name, "vectors": []}
        result[sid]["vectors"].append(json.loads(vec_json))
    return result


def delete_student(student_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()


def clear_roster():
    with get_conn() as conn:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM students")
        conn.commit()


# ---------- Sessions / Attendance history ----------

def create_session(total_roster, total_present, total_absent, summary, records):
    """records: list of {student_id, name, status, confidence}"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (created_at, total_roster, total_present, total_absent, summary) VALUES (?, ?, ?, ?, ?)",
            (time.time(), total_roster, total_present, total_absent, summary),
        )
        session_id = cur.lastrowid
        for r in records:
            conn.execute(
                "INSERT INTO attendance_records (session_id, student_id, name, status, confidence) VALUES (?, ?, ?, ?, ?)",
                (session_id, r.get("student_id"), r["name"], r["status"], r.get("confidence")),
            )
        conn.commit()
        return session_id


def get_history(limit=50):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, created_at, total_roster, total_present, total_absent, summary
            FROM sessions ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "id": r[0], "created_at": r[1], "total_roster": r[2],
            "total_present": r[3], "total_absent": r[4], "summary": r[5],
        }
        for r in rows
    ]


def get_session_detail(session_id: int):
    with get_conn() as conn:
        session = conn.execute("""
            SELECT id, created_at, total_roster, total_present, total_absent, summary
            FROM sessions WHERE id = ?
        """, (session_id,)).fetchone()
        records = conn.execute("""
            SELECT name, status, confidence FROM attendance_records WHERE session_id = ?
            ORDER BY status, name
        """, (session_id,)).fetchall()
    if not session:
        return None
    return {
        "id": session[0], "created_at": session[1], "total_roster": session[2],
        "total_present": session[3], "total_absent": session[4], "summary": session[5],
        "records": [{"name": r[0], "status": r[1], "confidence": r[2]} for r in records],
    }
