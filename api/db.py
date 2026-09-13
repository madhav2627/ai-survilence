"""
SURVILLENCE TRAFFIC — Persistent Database Layer
================================================
Unified server-side database adapter for Vercel deployment.
Strictly isolated per-user data store supporting:
  1. PostgreSQL (Vercel Postgres / Neon via POSTGRES_URL / DATABASE_URL)
  2. Serverless SQLite fallback (storage/traffic.db or /tmp/traffic_db/traffic.db)

Every record is scoped to `userId` — cross-user leakage is strictly impossible.
"""

import os
import json
import time
import uuid
import hashlib
import sqlite3
import threading
from pathlib import Path
from urllib.parse import urlparse

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = Path("/tmp") if Path("/tmp").exists() else (BASE_DIR / "storage")
DB_DIR = TMP_DIR / "traffic_db"
DB_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = DB_DIR / "traffic.db"

_lock = threading.Lock()

# ── Storage Backend Detection ───────────────────────────────────────────────

def _get_database_url() -> str | None:
    return os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")

def is_postgres() -> bool:
    return bool(_get_database_url())

def _get_pg_conn():
    url = _get_database_url()
    if not url:
        return None
    import ssl
    import pg8000.native
    u = urlparse(url)
    ctx = ssl.create_default_context()
    return pg8000.native.Connection(
        user=u.username,
        password=u.password,
        host=u.hostname,
        port=u.port or 5432,
        database=u.path.lstrip("/"),
        ssl_context=ctx
    )

def _get_sqlite_conn():
    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ── Unified Query Helpers ───────────────────────────────────────────────────

def _convert_placeholders(sql: str, params: tuple):
    parts = sql.split("?")
    if len(parts) <= 1:
        return sql, {}
    new_sql = ""
    param_dict = {}
    for i, part in enumerate(parts[:-1]):
        k = f"p{i}"
        new_sql += part + f":{k}"
        param_dict[k] = params[i]
    new_sql += parts[-1]
    return new_sql, param_dict

def _normalize_row(row_dict: dict) -> dict:
    if not row_dict:
        return row_dict
    d = dict(row_dict)
    if "full_name" in d and not d.get("name"):
        d["name"] = d["full_name"]
    if "name" in d and not d.get("full_name"):
        d["full_name"] = d["name"]
    if "id" in d:
        d["id"] = str(d["id"])
    return d

def _query_all(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        if is_postgres():
            try:
                con = _get_pg_conn()
                new_sql, kw = _convert_placeholders(sql, params)
                raw_rows = con.run(new_sql, **kw)
                if raw_rows and con.columns:
                    col_names = [c["name"] for c in con.columns]
                    results = [_normalize_row(dict(zip(col_names, r))) for r in raw_rows]
                else:
                    results = []
                con.close()
                return results
            except Exception as e:
                print(f"[DB] Postgres query_all error ({e}), falling back to SQLite", flush=True)

        conn = _get_sqlite_conn()
        try:
            cur = conn.execute(sql, params)
            return [_normalize_row(dict(r)) for r in cur.fetchall()]
        finally:
            conn.close()

def _query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = _query_all(sql, params)
    return rows[0] if rows else None

def _execute(sql: str, params: tuple = ()) -> int:
    with _lock:
        if is_postgres():
            try:
                con = _get_pg_conn()
                # Handle SQLite-specific INSERT OR REPLACE
                if "INSERT OR REPLACE INTO" in sql:
                    # e.g. INSERT OR REPLACE INTO table (cols) VALUES (?, ?) -> DELETE WHERE id = ? then INSERT
                    table = sql.split("INSERT OR REPLACE INTO")[1].split("(")[0].strip()
                    con.run(f"DELETE FROM {table} WHERE id = :p0", p0=str(params[0]))
                    mod_sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
                    new_sql, kw = _convert_placeholders(mod_sql, params)
                else:
                    new_sql, kw = _convert_placeholders(sql, params)

                con.run(new_sql, **kw)
                count = con.row_count or 0
                con.close()
                return count
            except Exception as e:
                print(f"[DB] Postgres execute error ({e}), falling back to SQLite", flush=True)

        conn = _get_sqlite_conn()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

# ── Database Initialization ─────────────────────────────────────────────────

def init_db():
    # 1. Initialize Postgres if configured
    if is_postgres():
        try:
            con = _get_pg_conn()
            con.run("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    full_name TEXT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    role TEXT DEFAULT 'operator',
                    settings TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_vehicles INTEGER DEFAULT 0,
                    cars INTEGER DEFAULT 0,
                    motorcycles INTEGER DEFAULT 0,
                    autos INTEGER DEFAULT 0,
                    buses INTEGER DEFAULT 0,
                    trucks INTEGER DEFAULT 0,
                    video_duration REAL DEFAULT 0,
                    processing_time REAL DEFAULT 0,
                    input_video TEXT,
                    output_video TEXT,
                    report_path TEXT,
                    video_available INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    stage TEXT DEFAULT 'Initializing',
                    original_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    notification_dismissed INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    camera_type TEXT NOT NULL,
                    stream_url TEXT NOT NULL,
                    username TEXT,
                    password TEXT,
                    port INTEGER,
                    status TEXT DEFAULT 'NOT CONFIGURED',
                    last_connected TEXT,
                    resolution TEXT,
                    fps REAL,
                    created_at TEXT NOT NULL
                );
            """)
            con.close()
            print("[DB] PostgreSQL / Neon initialized successfully.", flush=True)
        except Exception as exc:
            print(f"[DB] Could not initialize PostgreSQL: {exc}", flush=True)

    # 2. Also initialize SQLite as fallback
    with _lock:
        conn = _get_sqlite_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    role TEXT DEFAULT 'operator',
                    settings TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_vehicles INTEGER DEFAULT 0,
                    cars INTEGER DEFAULT 0,
                    motorcycles INTEGER DEFAULT 0,
                    autos INTEGER DEFAULT 0,
                    buses INTEGER DEFAULT 0,
                    trucks INTEGER DEFAULT 0,
                    video_duration REAL DEFAULT 0,
                    processing_time REAL DEFAULT 0,
                    input_video TEXT,
                    output_video TEXT,
                    report_path TEXT,
                    video_available INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    stage TEXT DEFAULT 'Initializing',
                    original_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    notification_dismissed INTEGER DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    camera_type TEXT NOT NULL,
                    stream_url TEXT NOT NULL,
                    username TEXT,
                    password TEXT,
                    port INTEGER,
                    status TEXT DEFAULT 'NOT CONFIGURED',
                    last_connected TEXT,
                    resolution TEXT,
                    fps REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        finally:
            conn.close()

# Initialize immediately
init_db()

# ── User Helpers ─────────────────────────────────────────────────────────────

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_user(name: str, username: str, email: str, password: str):
    name = (name or "").strip()
    username = (username or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not name:
        return None, "Full name is required.", 400
    if not email or "@" not in email:
        return None, "A valid email address is required.", 400
    if not username or len(username) < 3:
        return None, "Username must be at least 3 characters.", 400
    if not all(c.isalnum() or c in ("_", "-") for c in username):
        return None, "Username can only contain letters, numbers, hyphens, and underscores.", 400
    if len(password) < 6:
        return None, "Password must be at least 6 characters.", 400

    row = _query_one("SELECT id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    if row:
        return None, "Username is already taken.", 409
    row = _query_one("SELECT id FROM users WHERE LOWER(email) = LOWER(?)", (email,))
    if row:
        return None, "An account with this email already exists.", 409

    salt = uuid.uuid4().hex[:16]
    pw_hash = hash_password(password, salt)
    user_id = f"usr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    _execute(
        """
        INSERT INTO users (id, name, username, email, password_hash, salt, created_at, role, settings)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'operator', '{}')
        """,
        (user_id, name, username, email, pw_hash, salt, now_iso)
    )

    return {
        "id": user_id,
        "userId": user_id,
        "fullName": name,
        "name": name,
        "username": username,
        "email": email,
        "role": "operator",
        "createdAt": now_iso,
        "settings": {},
    }, None, 201

def authenticate_user(identifier: str, password: str):
    identifier = (identifier or "").strip()
    password = password or ""
    if not identifier:
        return None, "Email or username is required.", 400
    if not password:
        return None, "Password is required.", 400

    row = _query_one(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
        (identifier, identifier)
    )

    if not row:
        return None, "Account not found", 404

    pw_hash = row.get("password_hash") or ""
    salt = row.get("salt") or ""

    authenticated = False

    # 1. Support werkzeug-hashed passwords (e.g. scrypt / pbkdf2)
    if pw_hash.startswith("scrypt:") or pw_hash.startswith("pbkdf2:"):
        try:
            from werkzeug.security import check_password_hash
            authenticated = check_password_hash(pw_hash, password)
        except Exception:
            authenticated = False

    # 2. Support standard sha256 + salt
    if not authenticated:
        computed_hash = hash_password(password, salt)
        if computed_hash == pw_hash:
            authenticated = True

    if not authenticated:
        return None, "Incorrect password", 401

    try:
        settings = json.loads(row.get("settings") or "{}")
    except Exception:
        settings = {}

    user_name = row.get("name") or row.get("full_name") or row.get("username")
    return {
        "id": str(row["id"]),
        "userId": str(row["id"]),
        "fullName": user_name,
        "name": user_name,
        "username": row.get("username", ""),
        "email": row.get("email", ""),
        "role": row.get("role", "operator"),
        "createdAt": row.get("created_at", ""),
        "settings": settings,
    }, None, 200

def get_user(user_id: str):
    if not user_id:
        return None
    row = _query_one("SELECT * FROM users WHERE id = ?", (str(user_id),))
    if not row:
        return None
    try:
        settings = json.loads(row.get("settings") or "{}")
    except Exception:
        settings = {}
    user_name = row.get("name") or row.get("full_name") or row.get("username")
    return {
        "id": str(row["id"]),
        "userId": str(row["id"]),
        "fullName": user_name,
        "name": user_name,
        "username": row.get("username", ""),
        "email": row.get("email", ""),
        "role": row.get("role", "operator"),
        "createdAt": row.get("created_at", ""),
        "settings": settings,
    }

def update_user_settings(user_id: str, new_settings: dict) -> bool:
    if not user_id:
        return False
    row = _query_one("SELECT settings FROM users WHERE id = ?", (str(user_id),))
    if not row:
        return False
    try:
        curr = json.loads(row.get("settings") or "{}")
    except Exception:
        curr = {}
    curr.update(new_settings)
    _execute("UPDATE users SET settings = ? WHERE id = ?", (json.dumps(curr), str(user_id)))
    return True

def update_user_profile(user_id: str, full_name: str | None, email: str | None):
    if not user_id:
        return None, "Invalid user ID", 400
    row = _query_one("SELECT * FROM users WHERE id = ?", (str(user_id),))
    if not row:
        return None, "User not found", 404
    curr_name = row.get("name") or row.get("full_name") or row.get("username")
    new_name = full_name.strip() if full_name else curr_name
    new_email = email.strip().lower() if email else row.get("email", "")
    if new_email != row.get("email"):
        conflict = _query_one("SELECT id FROM users WHERE LOWER(email) = LOWER(?) AND id != ?", (new_email, str(user_id)))
        if conflict:
            return None, "Email is already in use by another account.", 409
    _execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (new_name, new_email, str(user_id)))
    return {
        "id": str(user_id),
        "userId": str(user_id),
        "fullName": new_name,
        "name": new_name,
        "username": row.get("username", ""),
        "email": new_email,
        "role": row.get("role", "operator"),
    }, None, 200

# ── Job Helpers ──────────────────────────────────────────────────────────────

def create_job(user_id: str, analysis_id: str, original_filename: str) -> dict:
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _execute(
        """
        INSERT INTO jobs (id, analysis_id, user_id, status, progress, stage, original_filename, created_at)
        VALUES (?, ?, ?, 'processing', 5, 'Upload received and validated', ?, ?)
        """,
        (job_id, analysis_id, str(user_id), original_filename, now_iso)
    )
    return {
        "jobId": job_id,
        "analysisId": analysis_id,
        "userId": str(user_id),
        "status": "processing",
        "progress": 5,
        "stage": "Upload received and validated",
        "originalFilename": original_filename,
        "createdAt": now_iso,
    }

def get_active_job(user_id: str) -> dict | None:
    if not user_id:
        return None
    row = _query_one(
        "SELECT * FROM jobs WHERE user_id = ? AND status = 'processing' ORDER BY created_at DESC LIMIT 1",
        (str(user_id),)
    )
    if not row:
        return None
    return {
        "jobId": row["id"],
        "sessionId": row.get("analysis_id"),
        "analysisId": row.get("analysis_id"),
        "userId": str(row["user_id"]),
        "status": row.get("status"),
        "progress": row.get("progress", 0),
        "stage": row.get("stage", ""),
        "filename": row.get("original_filename", ""),
        "createdAt": row.get("created_at", ""),
    }

def update_job_progress(job_id: str, progress: int, stage: str):
    _execute("UPDATE jobs SET progress = ?, stage = ? WHERE id = ?", (progress, stage, str(job_id)))

def complete_job(job_id: str, status: str = "completed", error: str = None):
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prog = 100 if status == "completed" else 0
    stage = "Analysis complete" if status == "completed" else f"Failed: {error or 'Unknown error'}"
    _execute(
        """
        UPDATE jobs SET status = ?, progress = ?, stage = ?, completed_at = ?, error = ?
        WHERE id = ?
        """,
        (status, prog, stage, now_iso, error, str(job_id))
    )

def get_recently_completed_job(user_id: str, ttl_seconds: int = 7200) -> dict | None:
    if not user_id:
        return None
    row = _query_one(
        """
        SELECT * FROM jobs
        WHERE user_id = ? AND status = 'completed' AND notification_dismissed = 0
        ORDER BY completed_at DESC LIMIT 1
        """,
        (str(user_id),)
    )
    if not row:
        return None
    return {
        "jobId": row["id"],
        "sessionId": row.get("analysis_id"),
        "analysisId": row.get("analysis_id"),
        "filename": row.get("original_filename", ""),
        "completedAt": row.get("completed_at", ""),
        "status": row.get("status", ""),
    }

def dismiss_completed_job(user_id: str, session_id: str):
    _execute("UPDATE jobs SET notification_dismissed = 1 WHERE user_id = ? AND analysis_id = ?", (str(user_id), str(session_id)))

# ── Analyses & History Helpers ──────────────────────────────────────────────

def add_analysis(user_id: str, analysis: dict):
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sid = analysis.get("sessionId") or analysis.get("id") or f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    filename = analysis.get("filename", "traffic_video.mp4")

    # Mark older physical videos as unavailable
    _execute("UPDATE analyses SET video_available = 0 WHERE user_id = ? AND id != ?", (str(user_id), sid))

    # Insert or replace analysis record
    _execute(
        """
        INSERT OR REPLACE INTO analyses (
            id, user_id, filename, status, total_vehicles, cars, motorcycles, autos, buses, trucks,
            video_duration, processing_time, input_video, output_video, report_path, video_available, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (
            sid, str(user_id), filename,
            analysis.get("status", "completed"),
            int(analysis.get("total_vehicles", 0)),
            int(analysis.get("cars", 0)),
            int(analysis.get("motorcycles", 0)),
            int(analysis.get("auto_rickshaws", analysis.get("autos", 0))),
            int(analysis.get("buses", 0)),
            int(analysis.get("trucks", 0)),
            float(analysis.get("video_duration", 0.0)),
            float(analysis.get("processing_time", 0.0)),
            str(analysis.get("input_video", "")),
            str(analysis.get("output_video", "")),
            str(analysis.get("report_path", "")),
            now_iso,
        )
    )

def mark_previous_videos_unavailable(user_id: str, keep_session_id: str | None = None) -> int:
    if not user_id:
        return 0
    if keep_session_id:
        return _execute("UPDATE analyses SET video_available = 0 WHERE user_id = ? AND id != ? AND video_available != 0", (str(user_id), keep_session_id))
    return _execute("UPDATE analyses SET video_available = 0 WHERE user_id = ? AND video_available != 0", (str(user_id),))

def get_user_history(user_id: str) -> list:
    if not user_id:
        return []
    rows = _query_all("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (str(user_id),))
    result = []
    for r in rows:
        result.append({
            "sessionId": r["id"],
            "session_id": r["id"],
            "id": r["id"],
            "userId": r["user_id"],
            "filename": r.get("filename", ""),
            "status": r.get("status", ""),
            "total_vehicles": r.get("total_vehicles", 0),
            "totalVehicles": r.get("total_vehicles", 0),
            "cars": r.get("cars", 0),
            "motorcycles": r.get("motorcycles", 0),
            "auto_rickshaws": r.get("autos", 0),
            "autos": r.get("autos", 0),
            "buses": r.get("buses", 0),
            "trucks": r.get("trucks", 0),
            "video_duration": r.get("video_duration", 0.0),
            "processing_time": r.get("processing_time", 0.0),
            "date": r.get("created_at", ""),
            "createdAt": r.get("created_at", ""),
            "videoAvailable": bool(r.get("video_available", 1)),
            "input_video": r.get("input_video", ""),
            "output_video": r.get("output_video", ""),
            "report_path": r.get("report_path", ""),
        })
    return result

def get_user_analysis(user_id: str, session_id: str) -> dict | None:
    if not user_id or not session_id:
        return None
    r = _query_one("SELECT * FROM analyses WHERE user_id = ? AND id = ?", (str(user_id), str(session_id)))
    if not r:
        return None
    return {
        "sessionId": r["id"],
        "session_id": r["id"],
        "userId": r["user_id"],
        "filename": r.get("filename", ""),
        "status": r.get("status", ""),
        "total_vehicles": r.get("total_vehicles", 0),
        "cars": r.get("cars", 0),
        "motorcycles": r.get("motorcycles", 0),
        "auto_rickshaws": r.get("autos", 0),
        "buses": r.get("buses", 0),
        "trucks": r.get("trucks", 0),
        "video_duration": r.get("video_duration", 0.0),
        "processing_time": r.get("processing_time", 0.0),
        "date": r.get("created_at", ""),
        "videoAvailable": bool(r.get("video_available", 1)),
        "input_video": r.get("input_video", ""),
        "output_video": r.get("output_video", ""),
        "report_path": r.get("report_path", ""),
    }

def delete_user_analysis(user_id: str, session_id: str) -> bool:
    if not user_id or not session_id:
        return False
    row = _query_one("SELECT * FROM analyses WHERE user_id = ? AND id = ?", (str(user_id), str(session_id)))
    if not row:
        return False
    for k in ("input_video", "output_video", "report_path"):
        p = row.get(k)
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except Exception:
                pass
    _execute("DELETE FROM analyses WHERE user_id = ? AND id = ?", (str(user_id), str(session_id)))
    _execute("DELETE FROM reports WHERE user_id = ? AND analysis_id = ?", (str(user_id), str(session_id)))
    _execute("DELETE FROM jobs WHERE user_id = ? AND analysis_id = ?", (str(user_id), str(session_id)))
    return True

# ── Report Helpers ───────────────────────────────────────────────────────────

def save_report(user_id: str, analysis_id: str, report_data: dict):
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_id = f"rep_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    _execute(
        """
        INSERT OR REPLACE INTO reports (id, analysis_id, user_id, report_data, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (report_id, str(analysis_id), str(user_id), json.dumps(report_data), now_iso)
    )

def get_report(user_id: str, session_id: str) -> dict | None:
    if not user_id or not session_id:
        return None
    row = _query_one("SELECT report_data FROM reports WHERE user_id = ? AND analysis_id = ?", (str(user_id), str(session_id)))
    if row and row.get("report_data"):
        try:
            return json.loads(row["report_data"])
        except Exception:
            pass
    return None

# ── Camera Helpers ───────────────────────────────────────────────────────────

def get_cameras(user_id: str) -> list:
    if not user_id:
        return []
    rows = _query_all("SELECT * FROM cameras WHERE user_id = ? ORDER BY created_at DESC", (str(user_id),))
    result = []
    for r in rows:
        result.append({
            "cameraId": r["id"],
            "id": r["id"],
            "userId": r["user_id"],
            "name": r.get("name", ""),
            "type": r.get("camera_type", ""),
            "streamUrl": r.get("stream_url", ""),
            "username": r.get("username", "") or "",
            "hasPassword": bool(r.get("password")),
            "port": r.get("port"),
            "status": r.get("status", "NOT CONFIGURED"),
            "lastConnected": r.get("last_connected"),
            "resolution": r.get("resolution"),
            "fps": r.get("fps"),
            "createdAt": r.get("created_at", ""),
        })
    return result

def get_camera(user_id: str, camera_id: str, include_password: bool = False) -> dict | None:
    if not user_id or not camera_id:
        return None
    r = _query_one("SELECT * FROM cameras WHERE id = ? AND user_id = ?", (str(camera_id), str(user_id)))
    if not r:
        return None
    res = {
        "cameraId": r["id"],
        "id": r["id"],
        "userId": r["user_id"],
        "name": r.get("name", ""),
        "type": r.get("camera_type", ""),
        "streamUrl": r.get("stream_url", ""),
        "username": r.get("username", "") or "",
        "hasPassword": bool(r.get("password")),
        "port": r.get("port"),
        "status": r.get("status", "NOT CONFIGURED"),
        "lastConnected": r.get("last_connected"),
        "resolution": r.get("resolution"),
        "fps": r.get("fps"),
        "createdAt": r.get("created_at", ""),
    }
    if include_password:
        res["password"] = r.get("password", "") or ""
    return res

def create_camera(user_id: str, name: str, camera_type: str, stream_url: str,
                  username: str = None, password: str = None, port: int = None):
    if not user_id:
        return None, "Unauthorized", 401
    name = (name or "").strip()
    if not name:
        return None, "Camera name is required.", 400
    camera_type = (camera_type or "IP Camera").strip()
    stream_url = (stream_url or "").strip()

    cam_id = f"cam_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    _execute(
        """
        INSERT INTO cameras (id, user_id, name, camera_type, stream_url, username, password, port, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'NOT CONFIGURED', ?)
        """,
        (cam_id, str(user_id), name, camera_type, stream_url, username or "", password or "", int(port) if port else None, now_iso)
    )
    return get_camera(user_id, cam_id), None, 201

def update_camera(user_id: str, camera_id: str, data: dict):
    if not user_id or not camera_id:
        return None, "Invalid request", 400
    row = _query_one("SELECT * FROM cameras WHERE id = ? AND user_id = ?", (str(camera_id), str(user_id)))
    if not row:
        return None, "Camera not found or access denied", 404
    name = data.get("name") if data.get("name") is not None else row.get("name")
    camera_type = data.get("type") if data.get("type") is not None else row.get("camera_type")
    stream_url = data.get("streamUrl") if data.get("streamUrl") is not None else row.get("stream_url")
    username = data.get("username") if data.get("username") is not None else row.get("username")
    pw_val = data.get("password")
    password = pw_val if (pw_val is not None and pw_val != "") else row.get("password")
    port = data.get("port") if data.get("port") is not None else row.get("port")
    status = data.get("status") if data.get("status") is not None else row.get("status")
    last_connected = data.get("lastConnected") if data.get("lastConnected") is not None else row.get("last_connected")
    resolution = data.get("resolution") if data.get("resolution") is not None else row.get("resolution")
    fps = data.get("fps") if data.get("fps") is not None else row.get("fps")

    _execute(
        """
        UPDATE cameras SET name = ?, camera_type = ?, stream_url = ?, username = ?, password = ?, port = ?,
                           status = ?, last_connected = ?, resolution = ?, fps = ?
        WHERE id = ? AND user_id = ?
        """,
        (name, camera_type, stream_url, username, password, port, status, last_connected, resolution, fps, str(camera_id), str(user_id))
    )
    return get_camera(user_id, camera_id), None, 200

def delete_camera(user_id: str, camera_id: str) -> bool:
    if not user_id or not camera_id:
        return False
    cnt = _execute("DELETE FROM cameras WHERE id = ? AND user_id = ?", (str(camera_id), str(user_id)))
    return cnt > 0
