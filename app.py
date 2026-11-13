from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, session

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("FACE_LOGIN_DB", ROOT / "data" / "auth.db"))
FACE_SIZE = (160, 160)
REGISTRATION_SAMPLES = 12
MIN_LOGIN_SAMPLES = 4
MAX_LOGIN_SAMPLES = 8
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "70"))

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # Loopback HTTP only.
    PERMANENT_SESSION_LIFETIME=1800,
    MAX_CONTENT_LENGTH=3 * 1024 * 1024,
)

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(str(CASCADE_PATH))
if FACE_DETECTOR.empty():
    raise RuntimeError(f"Could not load OpenCV face detector: {CASCADE_PATH}")

login_attempts: dict[str, deque[float]] = defaultdict(deque)
ATTEMPT_WINDOW_SECONDS = 60
MAX_ATTEMPTS_PER_WINDOW = 6


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with closing(connect_db()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                full_name TEXT NOT NULL,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                face_image BLOB NOT NULL
            );
            """
        )
        db.commit()


def json_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def request_json() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("Send this request as JSON.")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("The request body must be a JSON object.")
    return data


def password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64
    )


def normalize_face(data_url: Any) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("A camera image is missing.")
    header, encoded = data_url.split(",", 1)
    if header not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise ValueError("Camera images must be JPEG or PNG.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("A camera image could not be decoded.") from exc
    if not raw or len(raw) > 1_200_000:
        raise ValueError("A camera image is empty or too large.")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("A camera image could not be read.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = FACE_DETECTOR.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(detections) == 0:
        raise ValueError("No face found. Move closer and face the camera.")
    if len(detections) != 1:
        raise ValueError("Only one face should be visible in each camera frame.")

    x, y, width, height = max(detections, key=lambda box: box[2] * box[3])
    crop = cv2.equalizeHist(gray[y : y + height, x : x + width])
    return cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)


def encode_face(face: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not save a face sample.")
    return encoded.tobytes()


def decode_samples(items: Any, minimum: int, maximum: int) -> list[np.ndarray]:
    if not isinstance(items, list) or not minimum <= len(items) <= maximum:
        raise ValueError(f"Capture between {minimum} and {maximum} camera frames.")
    return [normalize_face(item) for item in items]


def check_login_throttle() -> bool:
    now = time.monotonic()
    bucket = login_attempts[request.remote_addr or "unknown"]
    while bucket and now - bucket[0] > ATTEMPT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= MAX_ATTEMPTS_PER_WINDOW:
        return False
    bucket.append(now)
    return True


def make_registry() -> tuple[Any, dict[int, str]]:
    with closing(connect_db()) as db:
        rows = db.execute(
            """
            SELECT accounts.id, accounts.username, face_samples.face_image
            FROM accounts JOIN face_samples ON face_samples.account_id = accounts.id
            ORDER BY accounts.id, face_samples.id
            """
        ).fetchall()

    images: list[np.ndarray] = []
    labels: list[int] = []
    usernames: dict[int, str] = {}
    for row in rows:
        decoded = cv2.imdecode(
            np.frombuffer(row["face_image"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if decoded is None:
            continue
        images.append(cv2.resize(decoded, FACE_SIZE, interpolation=cv2.INTER_AREA))
        labels.append(int(row["id"]))
        usernames[int(row["id"])] = row["username"]

    if not images:
        raise ValueError("No registered face profiles are available.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer, usernames


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/me")
def current_user():
    user_id = session.get("account_id")
    if user_id is None:
        return jsonify({"ok": True, "authenticated": False})
    with closing(connect_db()) as db:
        row = db.execute(
            "SELECT username, full_name FROM accounts WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        session.clear()
        return jsonify({"ok": True, "authenticated": False})
    return jsonify(
        {
            "ok": True,
            "authenticated": True,
            "user": {"username": row["username"], "full_name": row["full_name"]},
        }
    )


@app.post("/api/register")
def register():
    try:
        data = request_json()
        username = str(data.get("username", "")).strip().lower()
        full_name = str(data.get("full_name", "")).strip()
        password = data.get("password", "")
        if not re.fullmatch(r"[a-z0-9_.-]{3,32}", username):
            return json_error(
                "Username must be 3–32 characters using letters, numbers, dots, dashes, or underscores.",
                400,
            )
        if not full_name or len(full_name) > 60:
            return json_error("Enter a name of 1–60 characters.", 400)
        if not isinstance(password, str) or len(password) < 10 or len(password) > 128:
            return json_error("Use a password between 10 and 128 characters.", 400)

        samples = decode_samples(
            data.get("frames"), REGISTRATION_SAMPLES, REGISTRATION_SAMPLES
        )
        salt = secrets.token_bytes(16)
        digest = password_digest(password, salt)
        with closing(connect_db()) as db:
            try:
                cursor = db.execute(
                    """
                    INSERT INTO accounts(username, full_name, password_salt, password_hash)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, full_name, salt, digest),
                )
            except sqlite3.IntegrityError:
                db.rollback()
                return json_error("That username is already registered.", 409)

            account_id = int(cursor.lastrowid)
            db.executemany(
                "INSERT INTO face_samples(account_id, face_image) VALUES (?, ?)",
                [(account_id, encode_face(face)) for face in samples],
            )
            db.commit()

        return jsonify(
            {
                "ok": True,
                "message": "Account created. Sign in with your password and face.",
            }
        ), 201
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Registration failed")
        return json_error("Registration could not be completed.", 500)


@app.post("/api/login")
def login():
    if not check_login_throttle():
        return json_error("Too many sign-in attempts. Wait one minute and try again.", 429)

    try:
        data = request_json()
        username = str(data.get("username", "")).strip().lower()
        password = data.get("password", "")
        if not username or not isinstance(password, str):
            return json_error("Enter your username and password.", 400)

        samples = decode_samples(data.get("frames"), MIN_LOGIN_SAMPLES, MAX_LOGIN_SAMPLES)
        with closing(connect_db()) as db:
            account = db.execute(
                """
                SELECT id, username, full_name, password_salt, password_hash
                FROM accounts WHERE username = ?
                """,
                (username,),
            ).fetchone()

        if account is None:
            return json_error("Sign-in failed. Check your details and try again.", 401)
        candidate = password_digest(password, account["password_salt"])
        if not hmac.compare_digest(candidate, account["password_hash"]):
            return json_error("Sign-in failed. Check your details and try again.", 401)

        recognizer, known_users = make_registry()
        account_id = int(account["id"])
        if account_id not in known_users:
            return json_error("No face profile is available for this account.", 401)

        matched = 0
        for face in samples:
            predicted_id, distance = recognizer.predict(face)
            if predicted_id == account_id and distance <= MATCH_THRESHOLD:
                matched += 1

        if matched < math.ceil(len(samples) * 0.6):
            return json_error("Sign-in failed. Check your details and try again.", 401)

        session.clear()
        session.permanent = True
        session["account_id"] = account_id
        login_attempts[request.remote_addr or "unknown"].clear()
        return jsonify(
            {
                "ok": True,
                "authenticated": True,
                "user": {"username": account["username"], "full_name": account["full_name"]},
            }
        )
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Sign-in failed")
        return json_error("Sign-in could not be completed.", 500)


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True, "authenticated": False})


@app.errorhandler(413)
def request_too_large(_error):
    return json_error("The camera upload is too large. Try again in better lighting.", 413)


init_db()

if __name__ == "__main__":
    # Loopback only; this demo is not hardened for network deployment.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
