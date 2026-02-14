"""
SYNC LOCAL DATABASE TO PRODUCTION
==================================
Run this after analyzing races locally to push your database to production.

Usage: python sync_db_to_production.py
"""

import os
import sqlite3
import subprocess
import sys

DB_FILE = "gold_high_iq.db"


def get_db_stats():
    """Get current local database stats."""
    if not os.path.exists(DB_FILE):
        print("❌ gold_high_iq.db not found locally!")
        return None

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    stats = {}
    for table in ["races_analyzed", "horses_analyzed", "gold_high_iq", "race_results"]:
        try:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = c.fetchone()[0]
        except Exception:
            stats[table] = 0

    conn.close()
    return stats


def sync():
    """Commit and push the database to GitHub → triggers Render redeploy."""
    stats = get_db_stats()
    if not stats:
        sys.exit(1)

    print("📊 Local database status:")
    print(f"   Races analyzed: {stats['races_analyzed']}")
    print(f"   Horses analyzed: {stats['horses_analyzed']}")
    print(f"   Gold entries: {stats['gold_high_iq']}")
    print(f"   Race results: {stats['race_results']}")
    print(f"   DB size: {os.path.getsize(DB_FILE) / 1024:.1f} KB")
    print()

    # Check if there are changes to push
    result = subprocess.run(
        ["git", "diff", "--name-only", DB_FILE], capture_output=True, text=True
    )

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", DB_FILE],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip() and not staged.stdout.strip():
        print("✅ Database is already in sync with GitHub (no changes).")
        return

    # Commit and push
    msg = (
        f"Sync DB: {stats['races_analyzed']} races, "
        f"{stats['horses_analyzed']} horses, "
        f"{stats['gold_high_iq']} gold entries"
    )

    print("🔄 Syncing to production...")
    subprocess.run(["git", "add", DB_FILE], check=True)
    subprocess.run(["git", "commit", "--no-verify", "-m", msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

    print()
    print("✅ Database pushed to GitHub!")
    print("🚀 Render will auto-redeploy in ~60 seconds.")
    print("🌐 Production site will have updated data at handicappinghorseraces.org")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sync()
