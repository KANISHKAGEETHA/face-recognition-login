# Face Recognition Login

A Flask web application for account registration and sign-in using a password and webcam face verification. Face samples and account records are stored in SQLite.

## Features

- Register an account with a username, full name, password, and 12 webcam frames.
- Sign in with a password and 4–8 webcam frames, matched against registered faces using LBPH.
- Detect and crop a single face per frame with OpenCV's Haar cascade.
- Store password hashes using scrypt and manage login sessions with Flask.
- Limit sign-in attempts to 6 per minute per client address.

## Technologies

- Python
- Flask
- OpenCV (opencv-contrib-python; LBPH face recognizer)
- NumPy
- SQLite

## Requirements

- Python 3.10 or newer
- A working webcam
- The project web page template at `templates/index.html` (the Flask app renders this at `/`)

## Setup

```bash
git clone https://github.com/KANISHKAGEETHA/face-recognition-login.git
cd face-recognition-login
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in a browser and allow camera access.

## Configuration

The app supports these environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | Flask session signing key | Random key generated at startup |
| `FACE_LOGIN_DB` | SQLite database file path | `data/auth.db` |
| `FACE_MATCH_THRESHOLD` | Maximum LBPH distance accepted as a match | `70` |
| `PORT` | Local server port | `5000` |

For repeatable sessions, set `FLASK_SECRET_KEY` to a persistent random value before starting the app. The default random key logs users out when the server restarts.

## API

- `GET /api/me` — check the current session.
- `POST /api/register` — register an account. JSON fields: `username`, `full_name`, `password`, and `frames` (12 JPEG/PNG data URLs).
- `POST /api/login` — sign in. JSON fields: `username`, `password`, and `frames` (4–8 JPEG/PNG data URLs).
- `POST /api/logout` — clear the current session.

## Notes

The app binds to `127.0.0.1` and is intended for local/demo use. Face matching accuracy depends on lighting, camera quality, and the configured threshold. The SQLite database contains password hashes and stored face samples; protect the `data/` directory and do not commit it.
