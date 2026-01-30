import sqlite3

from fastapi import Request
from app.config import settings

_conn: sqlite3.Connection | None = None

def get_db(request: Request) -> sqlite3.Connection:
    conn = getattr(request.app.state, "db", None)
    if conn is None:
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row
        request.app.state.db = conn
    return conn


