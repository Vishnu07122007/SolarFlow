# wsgi.py — gunicorn / Render / PythonAnywhere entrypoint
# Render start: gunicorn wsgi:application --bind 0.0.0.0:$PORT

import os
import sys

_PROJECT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT, ".env"))
except Exception:
    pass

from app import app as application  # noqa: E402

# Alias used by some hosts
app = application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
