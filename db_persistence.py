"""
DATABASE PERSISTENCE LAYER
===========================

Ensures gold_high_iq.db survives Render redeployments.

PROBLEM: Render's filesystem is ephemeral. Every code push / redeploy wipes
all local files including the SQLite database — destroying all learned weights,
race history, calibration data, and intelligent learning insights.

SOLUTION (layered):
  1. PRIMARY:  Render Persistent Disk (/data mount) — survives redeploys
  2. SECONDARY: GitHub API backup/restore — survives EVERYTHING
  3. FALLBACK:  Local filesystem (development / no persistence configured)

SETUP:
  Render Dashboard → Environment → Add these env vars:
    GITHUB_TOKEN  = ghp_xxxx...  (Personal Access Token with 'repo' scope)
    GITHUB_REPO   = craigstephens859-prog/Horse-Race-Ready

  Render Dashboard → Disks → Add disk:
    Name: race-data
    Mount Path: /data
    Size: 1 GB

Author: PhD-Level ML Infrastructure Engineer
Date: February 6, 2026
"""

import os
import shutil
import sqlite3
import json
import base64
import logging
import threading
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ===================== ENVIRONMENT DETECTION =====================

def is_render() -> bool:
    """Detect if running on Render.com."""
    return bool(os.environ.get('RENDER'))


def has_persistent_disk() -> bool:
    """Check if Render persistent disk is mounted and writable."""
    render_data_dir = "/data"
    return (os.path.isdir(render_data_dir) and
            os.access(render_data_dir, os.W_OK))


def get_persistent_db_path(db_name: str = "gold_high_iq.db") -> str:
    """
    Get the correct database path based on environment.

    Priority:
      1. Render persistent disk (/data/) — survives redeploys
      2. Local filesystem — development or ephemeral Render

    Returns:
        Absolute path to the database file
    """
    if is_render() and has_persistent_disk():
        db_path = os.path.join("/data", db_name)
        logger.info(f"🔒 Using Render persistent disk: {db_path}")
        return db_path

    if is_render():
        logger.warning(
            "⚠️ Render detected but /data not mounted! "
            "Database will NOT survive redeploys. "
            "Add a persistent disk in Render Dashboard → Disks."
        )

    # Local development or no persistent disk
    return db_name


# ===================== STARTUP: RESTORE / SEED =====================

def initialize_persistent_db(db_name: str = "gold_high_iq.db") -> str:
    """
    Master initialization: determine path, restore from backup if needed.

    Called once at app startup.

    Returns:
        The database path to use for all operations.
    """
    db_path = get_persistent_db_path(db_name)

    # If persistent disk path, ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    db_exists = os.path.exists(db_path) and os.path.getsize(db_path) > 0
    db_has_data = False

    if db_exists:
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [r[0] for r in cursor.fetchall()]
            if 'races_analyzed' in tables:
                cursor.execute("SELECT COUNT(*) FROM races_analyzed")
                db_has_data = cursor.fetchone()[0] > 0
            conn.close()
        except Exception:
            pass

    if db_has_data:
        logger.info(f"✅ Persistent DB has data: {db_path}")
        return db_path

    # DB is empty or missing — try to restore from GitHub backup
    restored = restore_from_github(db_path)
    if restored:
        logger.info("✅ Restored database from GitHub backup!")
        return db_path

    # Try to seed from a local copy (if persistent disk but repo has seed)
    seed_path = db_name  # In the repo working directory
    if (db_path != seed_path and
            os.path.exists(seed_path) and
            os.path.getsize(seed_path) > 0):
        logger.info(f"📦 Seeding persistent DB from repo: {seed_path} → {db_path}")
        shutil.copy2(seed_path, db_path)
        return db_path

    logger.info(f"📁 Fresh database will be created: {db_path}")
    return db_path


# ===================== GITHUB BACKUP / RESTORE =====================

def _get_github_config() -> tuple:
    """Get GitHub token and repo from environment."""
    token = os.environ.get('GITHUB_TOKEN', '')
    repo = os.environ.get('GITHUB_REPO', '')
    return token, repo


def _github_api_request(url: str, method: str = 'GET',
                        data: dict = None, token: str = '') -> Optional[dict]:
    """Make a GitHub API request."""
    import urllib.request
    import urllib.error

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "HorseRaceReady-App"
    }
    if data:
        headers["Content-Type"] = "application/json"

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.warning(f"GitHub API error {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        logger.warning(f"GitHub API request failed: {e}")
        return None


def backup_to_github(db_path: str) -> bool:
    """
    Backup critical database tables to GitHub as JSON.

    Stores as `backups/db_backup.json` on the `db-backup` branch.
    This survives Render redeployments AND disk failures.

    Requires env vars: GITHUB_TOKEN, GITHUB_REPO
    """
    token, repo = _get_github_config()
    if not token or not repo:
        logger.debug("GitHub backup skipped: GITHUB_TOKEN or GITHUB_REPO not set")
        return False

    try:
        # Export critical tables to JSON
        backup_data = _export_tables_to_json(db_path)
        if not backup_data:
            return False

        content = json.dumps(backup_data, indent=2, default=str)
        content_b64 = base64.b64encode(content.encode()).decode()

        branch = "db-backup"
        file_path = "backups/db_backup.json"
        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"

        # Ensure branch exists
        _ensure_branch_exists(repo, token, branch)

        # Get current file SHA (needed for updates)
        existing = _github_api_request(
            f"{api_url}?ref={branch}", token=token
        )
        sha = existing.get('sha') if existing else None

        # Create/update file
        payload = {
            "message": (
                f"🧠 Auto-backup: {backup_data.get('stats', {}).get('total_races', 0)} races, "
                f"{backup_data.get('stats', {}).get('total_horses', 0)} horses | "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ),
            "content": content_b64,
            "branch": branch
        }
        if sha:
            payload["sha"] = sha

        result = _github_api_request(api_url, method='PUT',
                                     data=payload, token=token)

        if result:
            races = backup_data.get('stats', {}).get('total_races', 0)
            logger.info(f"✅ GitHub backup complete: {races} races saved to {repo}/{branch}")
            return True
        return False

    except Exception as e:
        logger.warning(f"GitHub backup failed: {e}")
        return False


def backup_to_github_async(db_path: str):
    """Non-blocking backup — runs in background thread."""
    thread = threading.Thread(
        target=backup_to_github,
        args=(db_path,),
        daemon=True
    )
    thread.start()


def restore_from_github(db_path: str) -> bool:
    """
    Restore database from GitHub backup if local DB is empty/missing.

    Downloads `backups/db_backup.json` from the `db-backup` branch
    and imports all tables.
    """
    token, repo = _get_github_config()
    if not token or not repo:
        logger.debug("GitHub restore skipped: GITHUB_TOKEN or GITHUB_REPO not set")
        return False

    try:
        branch = "db-backup"
        file_path = "backups/db_backup.json"
        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"

        result = _github_api_request(api_url, token=token)
        if not result or 'content' not in result:
            logger.info("No GitHub backup found to restore")
            return False

        content = base64.b64decode(result['content']).decode()
        backup_data = json.loads(content)

        races_count = backup_data.get('stats', {}).get('total_races', 0)
        logger.info(f"📥 Found GitHub backup with {races_count} races. Restoring...")

        _import_tables_from_json(db_path, backup_data)

        logger.info("✅ Database restored from GitHub backup!")
        return True

    except Exception as e:
        logger.warning(f"GitHub restore failed: {e}")
        return False


def _ensure_branch_exists(repo: str, token: str, branch: str):
    """Create the backup branch if it doesn't exist."""
    try:
        # Check if branch exists
        url = f"https://api.github.com/repos/{repo}/branches/{branch}"
        result = _github_api_request(url, token=token)
        if result:
            return  # Branch exists

        # Get default branch SHA
        url = f"https://api.github.com/repos/{repo}/git/refs/heads/main"
        main_ref = _github_api_request(url, token=token)
        if not main_ref:
            return

        sha = main_ref['object']['sha']

        # Create branch
        url = f"https://api.github.com/repos/{repo}/git/refs"
        _github_api_request(url, method='POST', data={
            "ref": f"refs/heads/{branch}",
            "sha": sha
        }, token=token)

        logger.info(f"✅ Created GitHub branch: {branch}")
    except Exception as e:
        logger.debug(f"Branch creation note: {e}")


# ===================== JSON EXPORT / IMPORT =====================

def _export_tables_to_json(db_path: str) -> dict:
    """Export all critical tables to a JSON-serializable dict."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # All tables that contain learned/user data
        tables_to_export = [
            'races_analyzed',
            'horses_analyzed',
            'gold_high_iq',
            'race_results_summary',
            'learned_weights',
            'calibration_history',
            'race_insights',
            'pattern_frequency',
            'odds_drift_learning',
            'learned_feature_flags',
            'race_results',
        ]

        backup = {
            'format_version': 2,
            'exported_at': datetime.now().isoformat(),
            'stats': {},
            'tables': {}
        }

        total_races = 0
        total_horses = 0

        for table in tables_to_export:
            try:
                cursor.execute(f"SELECT * FROM {table}")  # nosec
                rows = cursor.fetchall()
                backup['tables'][table] = [dict(row) for row in rows]

                if table == 'races_analyzed':
                    total_races = len(rows)
                if table == 'horses_analyzed':
                    total_horses = len(rows)
            except Exception:
                backup['tables'][table] = []

        backup['stats'] = {
            'total_races': total_races,
            'total_horses': total_horses,
            'tables_exported': len([t for t in backup['tables']
                                    if backup['tables'][t]]),
        }

        conn.close()
        return backup

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return {}


def _import_tables_from_json(db_path: str, backup: dict):
    """Import tables from a JSON backup dict into the database."""
    tables_data = backup.get('tables', {})
    if not tables_data:
        return

    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()

    imported_counts = {}

    for table_name, rows in tables_data.items():
        if not rows:
            continue

        # Get column info for this table
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")  # nosec
            columns = [col[1] for col in cursor.fetchall()]
            if not columns:
                continue
        except Exception:
            continue

        count = 0
        for row in rows:
            # Filter to only columns that exist in the table
            filtered = {k: v for k, v in row.items() if k in columns}
            if not filtered:
                continue

            cols = ', '.join(filtered.keys())
            placeholders = ', '.join(['?'] * len(filtered))

            try:
                cursor.execute(
                    f"INSERT OR IGNORE INTO {table_name} ({cols}) "  # nosec
                    f"VALUES ({placeholders})",
                    list(filtered.values())
                )
                count += 1
            except Exception as e:
                logger.debug(f"Import row skipped for {table_name}: {e}")

        imported_counts[table_name] = count

    conn.commit()
    conn.close()

    logger.info(f"📥 Import complete: {imported_counts}")


# ===================== CONVENIENCE =====================

def get_persistence_status(db_path: str) -> Dict:
    """Get a status report of the persistence layer."""
    token, repo = _get_github_config()

    status = {
        'is_render': is_render(),
        'has_persistent_disk': has_persistent_disk(),
        'db_path': db_path,
        'db_exists': os.path.exists(db_path),
        'db_size_mb': (os.path.getsize(db_path) / (1024 * 1024)
                       if os.path.exists(db_path) else 0),
        'github_backup_configured': bool(token and repo),
        'github_repo': repo if repo else 'NOT SET',
        'persistence_level': 'NONE',
    }

    if has_persistent_disk():
        status['persistence_level'] = '🔒 PERSISTENT DISK (survives redeploys)'
    elif token and repo:
        status['persistence_level'] = '☁️ GITHUB BACKUP (restored on redeploy)'
    elif is_render():
        status['persistence_level'] = '⚠️ EPHEMERAL (data lost on redeploy!)'
    else:
        status['persistence_level'] = '💻 LOCAL (development mode)'

    return status
