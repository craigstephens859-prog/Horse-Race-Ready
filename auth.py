"""
Authentication module for Horse Race Ready.
Provides sign-up/login with hashed passwords stored in SQLite.
Includes 30-day free trial tracking and $9.99/month subscription status.
Uses Render persistent disk when available so accounts survive redeploys.
"""

import hashlib
import logging
import os
import re
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


# ─── Database path (same logic as db_persistence.py) ───────────────────────
def _is_render() -> bool:
    return bool(os.getenv("RENDER"))


def _has_persistent_disk() -> bool:
    return os.path.isdir("/data")


def get_auth_db_path() -> str:
    """Return path to auth database. Uses /data/ on Render for persistence."""
    db_name = "auth_users.db"
    if _is_render() and _has_persistent_disk():
        return os.path.join("/data", db_name)
    return db_name


# ─── Password hashing (PBKDF2-SHA256, stdlib only — no extra deps) ─────────
_HASH_ITERATIONS = 260_000  # OWASP recommendation for PBKDF2-SHA256


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _HASH_ITERATIONS)
    return dk.hex(), salt.hex()


def _verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Verify a password against stored hash and salt."""
    salt = bytes.fromhex(stored_salt)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _HASH_ITERATIONS)
    return dk.hex() == stored_hash


# ─── Database init ─────────────────────────────────────────────────────────

# Subscription constants
FREE_TRIAL_DAYS = 30
MONTHLY_PRICE = 9.99


def init_auth_db() -> str:
    """Initialize the auth database. Returns the db path."""
    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                trial_start TEXT,
                subscription_status TEXT DEFAULT 'trial',
                subscription_expires TEXT,
                stripe_customer_id TEXT
            )
        """)
        # Safely add new columns for existing databases
        for col_def in [
            ("trial_start", "TEXT"),
            ("subscription_status", "TEXT DEFAULT 'trial'"),
            ("subscription_expires", "TEXT"),
            ("stripe_customer_id", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col_def[0]} {col_def[1]}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()
        conn.close()
        logger.info(f"✅ Auth database ready at {db_path}")
    except Exception as e:
        logger.error(f"❌ Auth database init failed: {e}")
    return db_path


# ─── Validation ────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _validate_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def _validate_password(password: str) -> str | None:
    """Returns error message or None if valid."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    return None


# ─── Sign Up ───────────────────────────────────────────────────────────────
def signup(name: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user. Returns (success, message).
    """
    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Please enter your name."
    if not _validate_email(email):
        return False, "Please enter a valid email address."
    pw_error = _validate_password(password)
    if pw_error:
        return False, pw_error

    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()

        # Check for existing email
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "An account with this email already exists. Please log in."

        # Hash password and insert
        pw_hash, pw_salt = _hash_password(password)
        now = datetime.now(UTC).isoformat()
        trial_expires = (
            datetime.now(UTC) + timedelta(days=FREE_TRIAL_DAYS)
        ).isoformat()
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, password_salt, created_at, trial_start, subscription_status, subscription_expires) VALUES (?, ?, ?, ?, ?, ?, 'trial', ?)",
            (name, email, pw_hash, pw_salt, now, now, trial_expires),
        )
        conn.commit()
        conn.close()
        logger.info(f"✅ New user registered: {email}")
        return True, f"Welcome, {name}! Your 30-day free trial is now active."

    except Exception as e:
        logger.error(f"❌ Signup error: {e}")
        return False, f"Registration error: {e}"


# ─── Login ─────────────────────────────────────────────────────────────────
def login(email: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Authenticate a user. Returns (success, message, user_dict_or_None).
    """
    email = email.strip().lower()
    if not email or not password:
        return False, "Please enter both email and password.", None

    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, email, password_hash, password_salt, is_active, trial_start, subscription_status, subscription_expires FROM users WHERE email = ?",
            (email,),
        )
        row = cursor.fetchone()

        if not row:
            conn.close()
            return False, "No account found with this email. Please sign up.", None

        (
            user_id,
            name,
            user_email,
            pw_hash,
            pw_salt,
            is_active,
            trial_start,
            sub_status,
            sub_expires,
        ) = row

        if not is_active:
            conn.close()
            return False, "This account has been deactivated.", None

        if not _verify_password(password, pw_hash, pw_salt):
            conn.close()
            return False, "Incorrect password. Please try again.", None

        # Update last login
        now = datetime.now(UTC).isoformat()
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))
        conn.commit()
        conn.close()

        user = {
            "id": user_id,
            "name": name,
            "email": user_email,
            "trial_start": trial_start,
            "subscription_status": sub_status or "trial",
            "subscription_expires": sub_expires,
        }
        logger.info(f"✅ User logged in: {email}")
        return True, f"Welcome back, {name}!", user

    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        return False, f"Login error: {e}", None


# ─── Admin: get all users (for your dashboard) ────────────────────────────
def get_all_users() -> list[dict]:
    """Return all registered users (for admin view)."""
    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, email, created_at, last_login, is_active FROM users ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "name": r[1],
                "email": r[2],
                "created_at": r[3],
                "last_login": r[4],
                "is_active": bool(r[5]),
            }
            for r in rows
        ]
    except Exception:
        return []


def get_user_count() -> int:
    """Return total registered user count."""
    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ─── Subscription helpers ──────────────────────────────────────────────────
def get_subscription_status(user_id: int) -> dict:
    """
    Returns subscription info for a user:
    - status: 'trial', 'active', 'expired', 'cancelled'
    - days_remaining: days left on trial or subscription
    - is_accessible: whether user can access the app right now
    """
    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT trial_start, subscription_status, subscription_expires FROM users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"status": "unknown", "days_remaining": 0, "is_accessible": False}

        trial_start, sub_status, sub_expires = row
        now = datetime.now(UTC)

        # Check trial status
        if sub_status in (None, "trial"):
            if trial_start:
                trial_end = datetime.fromisoformat(trial_start) + timedelta(
                    days=FREE_TRIAL_DAYS
                )
                days_left = (trial_end - now).days
                if days_left > 0:
                    return {
                        "status": "trial",
                        "days_remaining": days_left,
                        "is_accessible": True,
                    }
                else:
                    return {
                        "status": "trial_expired",
                        "days_remaining": 0,
                        "is_accessible": False,
                    }
            return {
                "status": "trial",
                "days_remaining": FREE_TRIAL_DAYS,
                "is_accessible": True,
            }

        # Check active subscription
        if sub_status == "active":
            if sub_expires:
                exp = datetime.fromisoformat(sub_expires)
                days_left = (exp - now).days
                return {
                    "status": "active",
                    "days_remaining": max(0, days_left),
                    "is_accessible": days_left > 0,
                }
            return {"status": "active", "days_remaining": 30, "is_accessible": True}

        return {
            "status": sub_status or "expired",
            "days_remaining": 0,
            "is_accessible": False,
        }

    except Exception as e:
        logger.error(f"Subscription check error: {e}")
        return {
            "status": "error",
            "days_remaining": 0,
            "is_accessible": True,
        }  # Fail open


def activate_subscription(user_id: int, stripe_customer_id: str = "") -> bool:
    """Mark a user's subscription as active (called after Stripe payment)."""
    db_path = get_auth_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        expires = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        conn.execute(
            "UPDATE users SET subscription_status = 'active', subscription_expires = ?, stripe_customer_id = ? WHERE id = ?",
            (expires, stripe_customer_id, user_id),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Subscription activation error: {e}")
        return False
