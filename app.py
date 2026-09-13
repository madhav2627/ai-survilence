"""
SURVILLENCE TRAFFIC — Unified AI Application
=============================================
Single unified entrypoint connecting:
  1. Frontend static files & single-page application routes
  2. Persistent user-isolated database (db.py)
  3. Real IISc UVH-26 YOLOv11-S model + ByteTrack vehicle tracking (detector.py)
  4. Complete REST API for Auth, Video Processing, History, Analytics, and Cameras

Runs locally via `python app.py` AND deploys directly to Vercel Serverless.
NO simulation. NO mock bounding boxes. 100% real AI model execution.
"""

import os
import sys
import json
import time
import uuid
import queue
import shutil
import hashlib
import hmac
import base64
import urllib.parse
import threading
from pathlib import Path

import requests as _requests
from flask import Flask, request, jsonify, Response, send_file, send_from_directory, abort
from flask_cors import CORS

# ── Vercel Blob configuration ────────────────────────────────────────────────
# Set BLOB_READ_WRITE_TOKEN in Vercel Dashboard → Project → Environment Variables.
# The token is NEVER sent to the browser.  Only the server uses it.
BLOB_READ_WRITE_TOKEN: str = os.environ.get(
    "BLOB_READ_WRITE_TOKEN",
    "vercel_blob_rw_SRlaRmWvR3ct3PuP_fMy1EWnffJIO1qI0zDuhP3G9sDbQsi"
)
# Maximum video size we allow through Blob (2 GB)
MAX_BLOB_VIDEO_BYTES: int = 2 * 1024 * 1024 * 1024
# Vercel Blob API root
_BLOB_API = "https://blob.vercel-storage.com"

# Setup Python paths
ROOT_DIR = Path(__file__).resolve().parent
API_DIR = ROOT_DIR / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
from detector import process_video_analysis, MODEL_PT, MODEL_ONNX, get_yolo_detector

# Setup storage directories (writable in /tmp on Vercel)
TMP_DIR = Path("/tmp") if Path("/tmp").exists() else (ROOT_DIR / "storage")
UPLOAD_DIR = TMP_DIR / "traffic_uploads"
RESULTS_DIR = TMP_DIR / "traffic_results"
REPORTS_DIR = TMP_DIR / "traffic_reports"

for d in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Initialize Flask with frontend static serving
# Initialize Flask with custom static serving
FRONTEND_DIR = ROOT_DIR / "frontend"
app = Flask(__name__, static_folder=None)
CORS(app, origins=["*"])

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Id"
    response.headers["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
    return response

# In-memory progress tracking for background jobs
_active_threads: dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()

def get_current_user_id() -> str | None:
    raw = (
        request.headers.get("X-User-Id")
        or request.args.get("user_id")
        or request.form.get("user_id")
    )
    if not raw and request.is_json:
        try:
            body = request.get_json(silent=True)
            if body and isinstance(body, dict):
                raw = body.get("user_id") or body.get("userId")
        except Exception:
            pass
    if not raw:
        return None
    cleaned = "".join(c for c in str(raw).strip() if c.isalnum() or c in ("_", "-"))
    return cleaned if cleaned else None

def safe_filename(name: str) -> str:
    name = Path(name).name
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")
    return "".join(c if c in keep else "_" for c in name)[:100]

# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND ROUTING (Clean URLs + Pages + Static Assets)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index_page():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/login")
@app.route("/login.html")
def login_page():
    return send_from_directory(str(FRONTEND_DIR), "login.html")

@app.route("/register")
@app.route("/register.html")
def register_page():
    return send_from_directory(str(FRONTEND_DIR), "register.html")

PAGE_MAP = {
    "dashboard": "dashboard.html",
    "surveillance": "surveillance.html",
    "cameras": "cameras.html",
    "live": "live.html",
    "analytics": "analytics.html",
    "traffic-ai": "traffic-ai.html",
    "history": "history.html",
    "reports": "reports.html",
    "settings": "settings.html",
}

for _route_name, _html_target in PAGE_MAP.items():
    def _create_page_handler(target_file):
        return lambda: send_from_directory(str(FRONTEND_DIR / "pages"), target_file)
    app.add_url_rule(f"/{_route_name}", f"view_{_route_name}", _create_page_handler(_html_target))
    app.add_url_rule(f"/{_route_name}.html", f"view_{_route_name}_html", _create_page_handler(_html_target))

@app.route("/background.png")
def serve_bg():
    return send_from_directory(str(FRONTEND_DIR), "background.png")

@app.route("/right_side.png")
def serve_right_side():
    return send_from_directory(str(FRONTEND_DIR), "right_side.png")

@app.route("/pages/<path:filename>")
def pages_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "pages"), filename)

@app.route("/css/<path:filename>")
def css_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)

@app.route("/js/<path:filename>")
def js_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)

@app.route("/assets/<path:filename>")
def assets_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
@app.route("/api/health")
def health():
    detector_info = get_yolo_detector()
    engine_name = detector_info[0] if detector_info else "unavailable"
    return jsonify({
        "status": "ok",
        "app": "SURVILLENCE TRAFFIC AI",
        "mode": "Online Serverless / Direct App",
        "engine": engine_name,
        "model_pt_exists": MODEL_PT.exists(),
        "model_onnx_exists": MODEL_ONNX.exists(),
        "detector": "IISc UVH-26 YOLOv11-S + ByteTrack",
        "database": "Persistent Isolated Storage",
        "max_physical_videos_per_user": 1,
    })

# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
@app.route("/api/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    name = data.get("fullName") or data.get("name")
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    user, error, code = db.create_user(name, username, email, password)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "user": user}), code

@app.route("/api/auth/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    identifier = data.get("identifier") or data.get("username") or data.get("email")
    password = data.get("password")

    user, error, code = db.authenticate_user(identifier, password)
    if error:
        return jsonify({"ok": False, "error": error, "code": "USER_NOT_FOUND" if code == 404 else "INVALID_PASSWORD"}), code
    return jsonify({"ok": True, "user": user}), code

@app.route("/api/auth/me", methods=["GET"])
@app.route("/api/me", methods=["GET"])
def auth_me():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"authenticated": False, "error": "Not authenticated"}), 401
    user = db.get_user(user_id)
    if not user:
        return jsonify({"authenticated": False, "error": "User does not exist"}), 401
    return jsonify({"authenticated": True, "user": user})

@app.route("/api/auth/settings", methods=["POST"])
def auth_settings():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    if db.update_user_settings(user_id, data):
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "User not found"}), 404

@app.route("/api/auth/profile", methods=["POST"])
def auth_profile():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    user, error, code = db.update_user_profile(user_id, data.get("fullName"), data.get("email"))
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "user": user})

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ANALYSIS PIPELINE (REAL MODEL EXECUTION)
# ══════════════════════════════════════════════════════════════════════════════

def _cleanup_tmp_storage(ttl_seconds: int = 3600):
    """Ensure /tmp doesn't accumulate orphaned or abandoned video files.
    - NEVER deletes reports, history JSON, user database, or model files.
    - Deletes temporary video files in UPLOAD_DIR or RESULTS_DIR older than ttl_seconds (default 1 hour)
      or 0-byte abandoned files.
    """
    now = time.time()
    for d in [UPLOAD_DIR, RESULTS_DIR]:
        try:
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        try:
                            file_age = now - p.stat().st_mtime
                            if file_age > ttl_seconds or p.stat().st_size == 0:
                                p.unlink()
                        except Exception:
                            pass
        except Exception:
            pass

def _remove_previous_physical_videos(user_id: str, keep_session_id: str) -> int:
    """Delete retained physical input/output videos and blobs from older analyses.

    Analysis history and reports are deliberately preserved; only the physical
    video files and temporary blobs are removed. The newest uploaded session is never touched.
    """
    removed = 0
    entries = db.get_user_history(user_id)
    for entry in entries:
        if entry.get("sessionId") == keep_session_id:
            continue
        for key in ("input_video", "output_video"):
            raw_path = entry.get(key)
            if not raw_path:
                continue
            if raw_path.startswith("http"):
                try:
                    if _delete_blob_object(raw_path):
                        removed += 1
                except Exception as exc:
                    print(f"[Storage] Could not delete previous blob {raw_path}: {exc}", flush=True)
            else:
                path = Path(raw_path)
                try:
                    if path.exists() and path.is_file():
                        path.unlink()
                        removed += 1
                except Exception as exc:
                    print(f"[Storage] Could not delete previous {key}: {path} ({exc})", flush=True)

    # Keep history rows but mark older physical videos as unavailable.
    db.mark_previous_videos_unavailable(user_id, keep_session_id)
    return removed

def _run_detection_worker(
    user_id: str,
    job_id: str,
    session_id: str,
    input_path: str,
    output_path: str,
    report_path: str,
    orig_name: str,
    input_blob_url: str = "",   # Vercel Blob URL of the uploaded input video (empty if legacy upload)
) -> dict:
    try:
        def on_progress(pct, stage):
            db.update_job_progress(job_id, pct, stage)

        # ── Real IISc UVH-26 YOLOv11-S + ByteTrack inference ─────────────────
        report = process_video_analysis(
            input_path=input_path,
            output_video_path=output_path,
            output_report_path=report_path,
            session_id=session_id,
            progress_callback=on_progress,
            stride=2,
            max_dim=1280
        )

        # ── Immediately delete temporary input video from /tmp ───────────────
        t_clean_0 = time.time()
        if Path(input_path).exists():
            try:
                Path(input_path).unlink()
                print(f"[Storage] Deleted temporary input video: {input_path}", flush=True)
            except Exception as exc:
                print(f"[Storage] Could not delete input video {input_path}: {exc}", flush=True)

        # ── Immediately delete temporary input blob from Vercel Blob ─────────
        if input_blob_url:
            _delete_blob_object(input_blob_url)
            print(f"[Blob] Deleted temporary input blob: {input_blob_url[:80]}", flush=True)
        t_cleanup = time.time() - t_clean_0
        print(f"[PERF] Cleanup: {t_cleanup:.3f}s", flush=True)

        # ── Upload processed output video to Vercel Blob for cross-instance durability ──
        final_video_target = output_path
        if _blob_available() and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            out_blob_pathname = f"traffic-users/{user_id}/{session_id}/tracked_{session_id}.mp4"
            print(f"[Blob] Uploading processed video to Vercel Blob: {out_blob_pathname}", flush=True)
            t_up0 = time.time()
            out_blob_url = _upload_to_blob(output_path, out_blob_pathname, content_type="video/mp4")
            if out_blob_url:
                final_video_target = out_blob_url
                print(f"[Blob] Output video uploaded successfully ({time.time() - t_up0:.2f}s): {out_blob_url[:80]}", flush=True)
                # Remove local file from /tmp to keep serverless disk clean
                try:
                    Path(output_path).unlink()
                    print(f"[Storage] Cleaned temporary output video from /tmp: {output_path}", flush=True)
                except Exception:
                    pass

        # Save analysis metadata and report to database
        analysis_data = {
            "sessionId": session_id,
            "filename": orig_name,
            "status": "completed",
            "total_vehicles": report.get("total_vehicles", 0),
            "cars": report.get("cars", 0),
            "motorcycles": report.get("motorcycles", 0),
            "auto_rickshaws": report.get("auto_rickshaws", 0),
            "autos": report.get("auto_rickshaws", 0),
            "buses": report.get("buses", 0),
            "trucks": report.get("trucks", 0),
            "video_duration": report.get("video_duration", 0),
            "processing_time": report.get("processing_time_seconds", 0),
            "input_video": "",
            "output_video": final_video_target,
            "report_path": report_path,
            "video_available": 1,
        }
        db.add_analysis(user_id, analysis_data)
        db.save_report(user_id, session_id, report)
        db.complete_job(job_id, status="completed")

        # ── Retention: delete older physical video files and blobs for this user ───
        _remove_previous_physical_videos(user_id, keep_session_id=session_id)

        return report

    except Exception as exc:
        print(f"[Detector Worker Error] {exc}", flush=True)
        # Clean up temporary input and output files on error
        for p in (input_path, output_path):
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass
        if input_blob_url:
            _delete_blob_object(input_blob_url)
        db.complete_job(job_id, status="failed", error=str(exc))
        db.add_analysis(user_id, {
            "sessionId": session_id,
            "filename": orig_name,
            "status": "failed",
            "total_vehicles": 0,
            "input_video": "",
            "output_video": "",
            "report_path": "",
            "video_available": 0,
        })
        raise
    finally:
        with _threads_lock:
            _active_threads.pop(job_id, None)

# ══════════════════════════════════════════════════════════════════════════════
# VERCEL BLOB — SECURE CLIENT UPLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _blob_available() -> bool:
    """Return True when BLOB_READ_WRITE_TOKEN is configured."""
    return bool(BLOB_READ_WRITE_TOKEN)


def _generate_client_token(pathname: str, valid_for_seconds: int = 600) -> dict | None:
    """Generate a scoped, short-lived client upload token using HMAC-SHA256.

    The secret BLOB_READ_WRITE_TOKEN is kept strictly server-side.
    Returns the direct upload URL and clientToken for the browser to PUT.
    """
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "") or BLOB_READ_WRITE_TOKEN
    if not token:
        return None
    try:
        parts = token.split("_")
        if len(parts) < 4:
            return None
        store_id = parts[3]

        valid_until = int((time.time() + valid_for_seconds) * 1000)
        payload_dict = {
            "pathname": pathname,
            "validUntil": valid_until,
        }
        payload_json = json.dumps(payload_dict, separators=(",", ":"))
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")

        # HMAC with rw_token as secret key
        h = hmac.new(token.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256)
        secured_key = h.hexdigest()

        combined = f"{secured_key}.{payload_b64}"
        combined_b64 = base64.b64encode(combined.encode("utf-8")).decode("utf-8")
        client_token = f"vercel_blob_client_{store_id}_{combined_b64}"

        upload_url = f"https://blob.vercel-storage.com/{pathname}"
        return {
            "url": upload_url,
            "clientToken": client_token,
            "access": "private",
        }
    except Exception as exc:
        print(f"[Blob] _generate_client_token error: {exc}", flush=True)
        return None


def _delete_blob_object(blob_url: str) -> bool:
    """Delete a single blob object by its URL. Best-effort, never raises."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "") or BLOB_READ_WRITE_TOKEN
    if not token or not blob_url:
        return False
    try:
        resp = _requests.post(
            f"{_BLOB_API}/delete",
            json={"urls": [blob_url]},
            headers={
                "Authorization": f"Bearer {token}",
                "x-api-version": "7",
            },
            timeout=15,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        print(f"[Blob] delete exception: {exc}", flush=True)
        return False


def _upload_to_blob(file_path: str, pathname: str, content_type: str = "video/mp4") -> str | None:
    """Upload a file to private Vercel Blob storage using the server's BLOB_READ_WRITE_TOKEN."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "") or BLOB_READ_WRITE_TOKEN
    if not token or not Path(file_path).exists():
        return None
    try:
        with open(file_path, "rb") as f:
            resp = _requests.put(
                f"{_BLOB_API}/{pathname}",
                data=f,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-api-version": "7",
                    "x-vercel-blob-access": "private",
                    "Content-Type": content_type,
                },
                timeout=180,
            )
        if resp.status_code in (200, 201):
            data = resp.json()
            return data.get("url") or data.get("downloadUrl")
        else:
            print(f"[Blob] Upload returned {resp.status_code}: {resp.text[:200]}", flush=True)
    except Exception as exc:
        print(f"[Blob] Upload failed: {exc}", flush=True)
    return None


def _stream_blob_video(blob_url: str, as_attachment: bool = False, filename: str = "tracked_video.mp4"):
    """Stream a private Vercel Blob video to the browser with HTTP Range support."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "") or BLOB_READ_WRITE_TOKEN
    req_headers = {}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    if "Range" in request.headers:
        req_headers["Range"] = request.headers["Range"]

    try:
        r = _requests.get(blob_url, headers=req_headers, stream=True, timeout=60)
    except Exception as exc:
        print(f"[Blob] Streaming proxy error: {exc}", flush=True)
        abort(502)

    resp_headers = {}
    for h in ("Content-Type", "Content-Range", "Content-Length", "Accept-Ranges"):
        if h in r.headers:
            resp_headers[h] = r.headers[h]
    if "Accept-Ranges" not in resp_headers:
        resp_headers["Accept-Ranges"] = "bytes"
    if as_attachment:
        resp_headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return Response(
        r.iter_content(chunk_size=128 * 1024),
        status=r.status_code,
        headers=resp_headers,
        content_type=r.headers.get("Content-Type", "video/mp4"),
    )


def _validate_blob_url_ownership(blob_url: str, user_id: str, session_id: str) -> bool:
    """Ensure the blob URL contains the expected user/session path prefix.

    Pattern enforced: traffic-users/<user_id>/<session_id>/
    This prevents one user from pointing /api/analyze at another user's blob.
    """
    expected_fragment = f"traffic-users/{user_id}/{session_id}/"
    return expected_fragment in blob_url or expected_fragment in urllib.parse.unquote(blob_url)


@app.route("/api/blob/upload-token", methods=["POST"])
def blob_upload_token():
    """Return a short-lived Vercel Blob client upload token for direct browser upload.

    The browser obtains this token, uses it to PUT the video directly to
    Vercel Blob CDN, then sends only the returned blob_url to /api/analyze.
    The BLOB_READ_WRITE_TOKEN secret is never exposed to the browser.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    if not _blob_available():
        return jsonify({
            "error": "Vercel Blob is not configured. Set BLOB_READ_WRITE_TOKEN environment variable.",
            "code": "BLOB_NOT_CONFIGURED",
        }), 503

    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    filesize = int(data.get("filesize", 0))

    if not filename:
        return jsonify({"error": "filename is required"}), 400

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'. Allowed: MP4, AVI, MOV, MKV, WebM"}), 400

    if filesize > MAX_BLOB_VIDEO_BYTES:
        return jsonify({"error": "File exceeds maximum allowed size (2 GB)"}), 400

    # Generate a unique, user-scoped session ID and blob pathname
    session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    safe_name = safe_filename(filename)
    # Path format: traffic-users/<user_id>/<session_id>/<safe_filename>
    pathname = f"traffic-users/{user_id}/{session_id}/{safe_name}"

    token_data = _generate_client_token(pathname, valid_for_seconds=600)
    if not token_data:
        return jsonify({"error": "Failed to generate Blob upload token. Please try again."}), 500

    # Return ONLY what the browser needs — never the master BLOB_READ_WRITE_TOKEN
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "sessionId": session_id,
        "pathname": pathname,
        "filename": safe_name,
        # url: where browser should PUT the file
        "url": token_data.get("url"),
        # clientToken: short-lived scoped token for browser PUT
        "clientToken": token_data.get("clientToken"),
        "access": token_data.get("access", "private"),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized: user_id is required"}), 401

    # Enforce 1 active job per user
    existing_job = db.get_active_job(user_id)
    if existing_job:
        return jsonify({
            "error": "An analysis is already in progress. Please wait for it to complete.",
            "code": "ALREADY_PROCESSING",
            "activeJob": existing_job,
        }), 409

    # ── Path A: New Vercel Blob flow (JSON body with blob_url) ───────────────
    # The browser uploads the video directly to Vercel Blob and sends only
    # lightweight metadata here.  Video bytes never pass through this function.
    json_body = request.get_json(silent=True)
    if json_body and json_body.get("blob_url"):
        blob_url: str = json_body["blob_url"]
        filename: str = json_body.get("filename", "video.mp4")
        session_id: str = json_body.get("session_id") or json_body.get("sessionId") or ""
        client_user_id: str = json_body.get("user_id", user_id)

        # Security: reject if client tries to use a different user's session
        if client_user_id != user_id:
            return jsonify({"error": "user_id mismatch"}), 403

        # Validate session_id format (must look like sess_<digits>_<hex>)
        if not session_id or not session_id.startswith("sess_"):
            return jsonify({"error": "Invalid or missing session_id"}), 400

        # Validate blob URL ownership — must contain traffic-users/<user_id>/<session_id>/
        if not _validate_blob_url_ownership(blob_url, user_id, session_id):
            return jsonify({"error": "Blob URL does not match your user/session"}), 403

        # Validate file extension
        orig_name = safe_filename(filename)
        ext = Path(orig_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported format '{ext}'"}), 400

        # Download blob from Vercel CDN to /tmp
        input_path = str(UPLOAD_DIR / f"{session_id}_{orig_name}")
        output_path = str(RESULTS_DIR / f"{session_id}_output.mp4")
        report_path = str(REPORTS_DIR / f"{session_id}_report.json")

        # Serverless disk protection: clean stale files before download
        _cleanup_tmp_storage()

        try:
            print(f"[Blob] Downloading {blob_url[:80]}... to {input_path}", flush=True)
            t_b0 = time.time()
            headers = {}
            token = os.environ.get("BLOB_READ_WRITE_TOKEN", "") or BLOB_READ_WRITE_TOKEN
            if token:
                headers["Authorization"] = f"Bearer {token}"
            with _requests.get(blob_url, headers=headers, stream=True, timeout=180) as r:
                r.raise_for_status()

                # Check Content-Length against available disk space
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    expected_bytes = int(cl)
                    _, _, free_b = shutil.disk_usage(str(TMP_DIR))
                    if expected_bytes > free_b:
                        _cleanup_tmp_storage()
                        _, _, free_b = shutil.disk_usage(str(TMP_DIR))
                        if expected_bytes > free_b:
                            return jsonify({
                                "error": f"Video size ({expected_bytes / 1_048_576:.1f} MB) exceeds available serverless disk space ({free_b / 1_048_576:.1f} MB). Please use a video clip under 450 MB."
                            }), 400

                total = 0
                t_f0 = time.time()
                with open(input_path, "wb") as fout:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fout.write(chunk)
                            total += len(chunk)
                t_file_create = time.time() - t_f0

            t_blob_download = time.time() - t_b0
            print(f"[PERF] Blob download: {t_blob_download:.3f}s ({total / 1_048_576:.2f} MB @ {(total / 1_048_576) / max(t_blob_download, 0.001):.2f} MB/s)", flush=True)
            print(f"[PERF] Temporary file creation: {t_file_create:.3f}s", flush=True)
        except Exception as exc:
            try:
                if Path(input_path).exists():
                    Path(input_path).unlink()
            except Exception:
                pass
            print(f"[Blob] Download error: {exc}", flush=True)
            return jsonify({"error": f"Failed to retrieve uploaded video from Blob: {exc}"}), 500

        if not Path(input_path).exists() or Path(input_path).stat().st_size == 0:
            return jsonify({"error": "Downloaded video is empty or missing"}), 400

        job = db.create_job(user_id, session_id, orig_name)

        # Run AI detection synchronously inside this HTTP request so Vercel keeps the
        # serverless process active with dedicated CPU for the full maxDuration=800s.
        # This completely eliminates the issue of background threads getting frozen.
        try:
            report = _run_detection_worker(
                user_id=user_id,
                job_id=job["jobId"],
                session_id=session_id,
                input_path=input_path,
                output_path=output_path,
                report_path=report_path,
                orig_name=orig_name,
                input_blob_url=blob_url,
            )
            return jsonify({
                "sessionId": session_id,
                "session_id": session_id,
                "jobId": job["jobId"],
                "status": "completed",
                "filename": orig_name,
                "stage": "Analysis complete!",
                "progress": 100,
                "userId": user_id,
                "report": report,
            })
        except Exception as exc:
            return jsonify({
                "sessionId": session_id,
                "session_id": session_id,
                "jobId": job["jobId"],
                "status": "failed",
                "error": str(exc),
            }), 500

    # ── Path B: Legacy multipart file upload (local dev / fallback) ──────────
    # This path allows the existing FormData upload to continue working locally.
    # On Vercel, videos >4.5 MB will still hit 413 on this path — that is
    # expected; the browser should always use the Blob path in production.
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use /api/blob/upload-token for large files."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'. Allowed: MP4, AVI, MOV, MKV, WebM"}), 400

    session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    orig_name = safe_filename(f.filename)

    input_path = str(UPLOAD_DIR / f"{session_id}_{orig_name}")
    output_path = str(RESULTS_DIR / f"{session_id}_output.mp4")
    report_path = str(REPORTS_DIR / f"{session_id}_report.json")

    try:
        f.save(input_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save video: {exc}"}), 500

    if not Path(input_path).exists() or Path(input_path).stat().st_size == 0:
        return jsonify({"error": "Uploaded video file is empty or missing"}), 400

    removed_previous = _remove_previous_physical_videos(user_id, session_id)
    print(f"[Storage] New session {session_id}: removed {removed_previous} previous physical video file(s)", flush=True)

    job = db.create_job(user_id, session_id, orig_name)

    try:
        report = _run_detection_worker(
            user_id=user_id,
            job_id=job["jobId"],
            session_id=session_id,
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            orig_name=orig_name,
        )
        return jsonify({
            "sessionId": session_id,
            "session_id": session_id,
            "jobId": job["jobId"],
            "status": "completed",
            "filename": orig_name,
            "stage": "Analysis complete!",
            "progress": 100,
            "userId": user_id,
            "report": report,
        })
    except Exception as exc:
        return jsonify({
            "sessionId": session_id,
            "session_id": session_id,
            "jobId": job["jobId"],
            "status": "failed",
            "error": str(exc),
        }), 500

@app.route("/api/active-job")
def active_job():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"job": None, "active": False})
    job = db.get_active_job(user_id)
    # Note: Do NOT do t.join() here — on Vercel each request is a new serverless
    # instance and the in-memory _active_threads dict is always empty.  All state
    # is persisted in the database by the worker thread, so a plain DB read is
    # sufficient and avoids unnecessarily blocking the polling request.
    return jsonify({"job": job, "active": bool(job)})

@app.route("/api/completed-job")
def completed_job():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"job": None})
    job = db.get_recently_completed_job(user_id)
    return jsonify({"job": job})

@app.route("/api/completed-job/<session_id>/dismiss", methods=["POST"])
def dismiss_completed_job(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False}), 401
    db.dismiss_completed_job(user_id, session_id)
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>")
def get_job_status(job_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    job = db.get_active_job(user_id)
    if job and job.get("jobId") == job_id:
        return jsonify(job)
    recent = db.get_recently_completed_job(user_id)
    if recent and recent.get("jobId") == job_id:
        return jsonify(recent)
    return jsonify({"status": "unknown"})

@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    job = db.get_active_job(user_id)
    if job and job.get("jobId") == job_id:
        db.complete_job(job_id, status="cancelled", error="Cancelled by user")
        return jsonify({"ok": True, "status": "cancelled"})
    return jsonify({"ok": False, "error": "Job not found or not active"}), 404

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS, REPORTS & DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/result/<session_id>")
def result(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        return jsonify({"error": "Analysis not found or access denied"}), 404

    report_data = db.get_report(user_id, session_id)
    if not report_data and analysis.get("report_path"):
        rp = Path(analysis["report_path"])
        if rp.exists():
            try:
                report_data = json.loads(rp.read_text("utf-8"))
            except Exception:
                pass

    return jsonify({
        "status": analysis["status"],
        "report": report_data,
        "sessionId": session_id,
        "userId": user_id,
        "videoAvailable": analysis.get("videoAvailable", False),
        "historyEntry": analysis,
    })

@app.route("/api/video/<session_id>")
@app.route("/api/stream/video/<session_id>")
def video(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)

    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        abort(404)

    out_path = analysis.get("output_video")
    if out_path:
        if out_path.startswith("http"):
            return _stream_blob_video(out_path, as_attachment=False)
        if Path(out_path).exists():
            return send_file(str(out_path), mimetype="video/mp4", conditional=True)

    candidates = list(RESULTS_DIR.glob(f"{session_id}*.*"))
    if candidates:
        return send_file(str(candidates[0]), mimetype="video/mp4", conditional=True)

    abort(404)

@app.route("/api/download/video/<session_id>")
def download_video(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)
    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        abort(404)
    out_path = analysis.get("output_video")
    if out_path:
        if out_path.startswith("http"):
            return _stream_blob_video(out_path, as_attachment=True, filename=f"tracked_{session_id}.mp4")
        if Path(out_path).exists():
            return send_file(str(out_path), as_attachment=True, download_name=f"tracked_{session_id}.mp4")
    candidates = list(RESULTS_DIR.glob(f"{session_id}*.*"))
    if candidates:
        return send_file(str(candidates[0]), as_attachment=True, download_name=f"tracked_{session_id}.mp4")
    abort(404)

@app.route("/api/video/<session_id>/cleanup", methods=["POST"])
def cleanup_video(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        return jsonify({"ok": False, "error": "Analysis not found"}), 404

    # Remove temporary output video file or blob
    out_path = analysis.get("output_video")
    if out_path:
        if out_path.startswith("http"):
            _delete_blob_object(out_path)
            print(f"[Blob] Safely cleaned up temporary output blob after download: {out_path[:80]}", flush=True)
        elif Path(out_path).exists():
            try:
                Path(out_path).unlink()
                print(f"[Storage] Safely cleaned up temporary output video after download: {out_path}", flush=True)
            except Exception as e:
                print(f"[Storage] Could not delete output video: {e}", flush=True)

    for p in RESULTS_DIR.glob(f"{session_id}*.*"):
        try:
            p.unlink()
        except Exception:
            pass

    db.mark_video_unavailable(user_id, session_id)
    return jsonify({"ok": True, "cleaned": True})

@app.route("/api/download/report/<session_id>")
def download_report(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)
    report_data = db.get_report(user_id, session_id)
    if not report_data:
        analysis = db.get_user_analysis(user_id, session_id)
        if analysis and analysis.get("report_path") and Path(analysis["report_path"]).exists():
            report_data = json.loads(Path(analysis["report_path"]).read_text("utf-8"))
    if report_data:
        return Response(
            json.dumps(report_data, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=report_{session_id}.json"}
        )
    abort(404)

# ══════════════════════════════════════════════════════════════════════════════
# HISTORY & STORAGE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/history")
def history():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([])
    entries = db.get_user_history(user_id)
    return jsonify(entries)

@app.route("/api/history/<session_id>", methods=["DELETE"])
def delete_history(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    ok = db.delete_user_analysis(user_id, session_id)
    if not ok:
        return jsonify({"error": "Not found or access denied"}), 404
    return jsonify({"status": "deleted", "sessionId": session_id})

@app.route("/api/storage/stats")
def storage_stats():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({
            "videos_in_bytes": 0,
            "videos_out_bytes": 0,
            "reports_bytes": 0,
            "total_bytes": 0,
            "history_count": 0,
            "retained_videos": 0,
        })
    entries = db.get_user_history(user_id)
    retained = sum(1 for e in entries if e.get("videoAvailable", False))

    in_size = sum(Path(e["input_video"]).stat().st_size for e in entries if e.get("input_video") and not e["input_video"].startswith("http") and Path(e["input_video"]).exists())
    out_size = sum(Path(e["output_video"]).stat().st_size for e in entries if e.get("output_video") and not e["output_video"].startswith("http") and Path(e["output_video"]).exists())
    rep_size = sum(Path(e["report_path"]).stat().st_size for e in entries if e.get("report_path") and not e["report_path"].startswith("http") and Path(e["report_path"]).exists())

    return jsonify({
        "videos_in_bytes": in_size,
        "videos_out_bytes": out_size,
        "reports_bytes": rep_size,
        "total_bytes": in_size + out_size + rep_size,
        "history_count": len(entries),
        "retained_videos": retained,
    })

@app.route("/api/storage/cleanup", methods=["POST"])
def storage_cleanup():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"removed_files": 0})
    entries = db.get_user_history(user_id)
    retained_outputs = {e.get("output_video") for e in entries if e.get("videoAvailable", False)}
    removed = 0

    for e in entries:
        if not e.get("videoAvailable", False):
            for k in ("input_video", "output_video"):
                p = e.get(k)
                if p and p not in retained_outputs:
                    if p.startswith("http"):
                        try:
                            if _delete_blob_object(p):
                                removed += 1
                        except Exception:
                            pass
                    elif Path(p).exists():
                        try:
                            Path(p).unlink()
                            removed += 1
                        except Exception:
                            pass
    return jsonify({"removed_files": removed})

# ══════════════════════════════════════════════════════════════════════════════
# CAMERAS & CCTV CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/cameras", methods=["GET"])
def cameras_list():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([]), 200
    cameras = db.get_cameras(user_id)
    return jsonify(cameras)

@app.route("/api/cameras", methods=["POST"])
def cameras_create():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    name = data.get("name") or data.get("cameraName")
    camera_type = data.get("type") or data.get("cameraType") or "IP Camera"
    stream_url = data.get("streamUrl") or data.get("url") or ""
    username = data.get("username")
    password = data.get("password")
    port = data.get("port")

    cam, error, code = db.create_camera(user_id, name, camera_type, stream_url, username, password, port)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "camera": cam}), code

@app.route("/api/cameras/<camera_id>", methods=["GET"])
def cameras_get(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id)
    if not cam:
        return jsonify({"error": "Camera not found or access denied"}), 404
    return jsonify(cam)

@app.route("/api/cameras/<camera_id>", methods=["PUT", "POST"])
def cameras_update(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    cam, error, code = db.update_camera(user_id, camera_id, data)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "camera": cam})

@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def cameras_delete(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    ok = db.delete_camera(user_id, camera_id)
    if not ok:
        return jsonify({"ok": False, "error": "Camera not found or unauthorized"}), 404
    return jsonify({"ok": True, "deleted": camera_id})

def _probe_camera_connection(stream_url: str):
    stream_url = (stream_url or "").strip()
    if not stream_url:
        return "NOT CONFIGURED", None, None, "No stream URL provided."

    private_prefixes = ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
                        "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
                        "172.28.", "172.29.", "172.30.", "172.31.", "127.", "localhost")
    for prefix in private_prefixes:
        if prefix in stream_url:
            return (
                "PRIVATE LAN / UNREACHABLE",
                None,
                None,
                f"Camera URL contains '{prefix}', which is on a private local network. "
                "Cloud-deployed servers cannot reach private LAN cameras unless routed through a public secure endpoint."
            )

    if not (stream_url.startswith("rtsp://") or stream_url.startswith("http://") or stream_url.startswith("https://")):
        return "UNSUPPORTED STREAM", None, None, "Stream URL must begin with rtsp://, http://, or https://"

    try:
        import cv2
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            return "OFFLINE", None, None, "Camera stream could not be opened from cloud."
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return "ONLINE", f"{w} × {h}", 25.0, None
        return "NO SIGNAL", None, None, "Stream opened but no frames received."
    except Exception as exc:
        return "CONNECTION ERROR", None, None, str(exc)

@app.route("/api/cameras/test", methods=["POST"])
def cameras_test_raw():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    stream_url = data.get("streamUrl") or data.get("url") or ""
    status, resolution, fps, error = _probe_camera_connection(stream_url)
    return jsonify({
        "status": status,
        "online": status == "ONLINE",
        "resolution": resolution,
        "fps": fps,
        "error": error,
    })

@app.route("/api/cameras/<camera_id>/test", methods=["POST"])
def cameras_test_saved(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id, include_password=True)
    if not cam:
        return jsonify({"ok": False, "error": "Camera not found or access denied"}), 404

    status, resolution, fps, error = _probe_camera_connection(cam.get("streamUrl"))
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    updates = {"status": status}
    if status == "ONLINE":
        updates["lastConnected"] = now_iso
        if resolution:
            updates["resolution"] = resolution
        if fps:
            updates["fps"] = fps

    updated_cam, _, _ = db.update_camera(user_id, camera_id, updates)
    return jsonify({
        "status": status,
        "online": status == "ONLINE",
        "resolution": resolution,
        "fps": fps,
        "error": error,
        "camera": updated_cam,
    })

@app.route("/api/cameras/<camera_id>/stats")
def camera_live_stats(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id)
    if not cam:
        return jsonify({"error": "Camera not found"}), 404

    return jsonify({
        "cameraId": camera_id,
        "cameraName": cam["name"],
        "cameraType": cam["type"],
        "stats": {
            "active_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "autos": 0,
            "buses": 0,
            "trucks": 0,
            "fps": cam.get("fps") or 25.0,
            "status": cam.get("status", "NOT CONFIGURED"),
        }
    })

# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"SURVILLENCE TRAFFIC AI — Running on http://0.0.0.0:{port}")
    print(f"Model PT:   {MODEL_PT} (exists: {MODEL_PT.exists()})")
    print(f"Model ONNX: {MODEL_ONNX} (exists: {MODEL_ONNX.exists()})")
    det = get_yolo_detector()
    print(f"Active Engine: {det[0] if det else 'None'}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
