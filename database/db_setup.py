"""
db_setup.py
Creates the SQLite database and seeds it with default evaluation criteria.
Run standalone:  python database/db_setup.py
Or imported by app.py to guarantee the DB exists before Streamlit starts.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "rfp_evaluation.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    description     TEXT,
    weight          REAL NOT NULL,      -- percentage, e.g. 30 means 30%
    max_score       INTEGER NOT NULL DEFAULT 10,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id      TEXT PRIMARY KEY,   -- UUID
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'created'
);

CREATE TABLE IF NOT EXISTS supplier_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id          TEXT NOT NULL,
    supplier_name       TEXT NOT NULL,
    submission_date     TEXT NOT NULL,
    experience_rating   REAL NOT NULL,
    absolute_score      REAL,
    ppi                 REAL,
    final_rank          INTEGER,
    result_json         TEXT,
    FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs (rfp_run_id)
);
"""

DEFAULT_CRITERIA = [
    ("Technical Capability", "Architecture, integrations, scalability, technical fit", 30, 10, 1),
    ("Implementation Plan", "Timeline, milestones, staffing, risk plan", 20, 10, 1),
    ("Commercial Value", "Pricing clarity, total cost, assumptions", 20, 10, 1),
    ("Security & Compliance", "Controls, certifications, privacy, auditability", 20, 10, 1),
    ("Support & Experience", "Support model, similar projects, references", 10, 10, 1),
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(reseed_if_empty=True):
    """Creates tables if they don't exist and seeds default criteria if the
    criteria table is empty. Safe to call every app startup."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()

    if reseed_if_empty:
        cur.execute("SELECT COUNT(*) AS c FROM evaluation_criteria")
        count = cur.fetchone()["c"]
        if count == 0:
            cur.executemany(
                """INSERT INTO evaluation_criteria
                   (name, description, weight, max_score, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                DEFAULT_CRITERIA,
            )
            conn.commit()
    conn.close()


def get_active_criteria():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def active_weight_total():
    return sum(c["weight"] for c in get_active_criteria())


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
    crit = get_active_criteria()
    print(f"{len(crit)} active criteria, total weight = {active_weight_total()}%")
    for c in crit:
        print(f"  - {c['name']} ({c['weight']}%)")
