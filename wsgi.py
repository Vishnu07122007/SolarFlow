# wsgi.py — PythonAnywhere / gunicorn entrypoint
# On PythonAnywhere: Web → WSGI configuration file → point to this file
# and ensure the project folder is on sys.path (see DEPLOY_PYTHONANYWHERE.md).

import os
import sys

# Project root = directory containing this file
_PROJECT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

# Load .env before importing app (app also loads it; this is belt-and-suspenders)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT, ".env"))
except Exception:
    pass

from app import app as application  # noqa: E402  — PA expects `application`

# Optional: local `python wsgi.py` smoke test
if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
