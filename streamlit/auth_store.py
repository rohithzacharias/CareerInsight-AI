"""Small local SQLite account store with salted PBKDF2 password hashes."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path


ITERATIONS = 310_000


def _connection(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    return connection


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def create_user(database_path: Path, name: str, email: str, password: str) -> tuple[bool, str]:
    if not name.strip() or "@" not in email or len(password) < 8:
        return False, "Enter a name, a valid email address, and a password of at least 8 characters."
    try:
        connection = _connection(database_path)
        try:
            connection.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name.strip(), email.strip().lower(), _hash_password(password)))
            connection.commit()
        finally:
            connection.close()
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "An account already exists for that email address."


def authenticate_user(database_path: Path, email: str, password: str) -> tuple[bool, str | None]:
    connection = _connection(database_path)
    try:
        row = connection.execute("SELECT name, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    finally:
        connection.close()
    if row is None:
        return False, None
    salt_hex, stored_digest = row[1].split("$", maxsplit=1)
    candidate = _hash_password(password, bytes.fromhex(salt_hex)).split("$", maxsplit=1)[1]
    return hmac.compare_digest(candidate, stored_digest), row[0]
