"""
Quick database consistency check for Gold High-IQ System
"""

import sqlite3

DB_FILE = "gold_high_iq.db"

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

print("=" * 60)
print("DATABASE CONSISTENCY CHECK")
print("=" * 60)

# Basic counts
print("\n📊 TABLE COUNTS:")
tables = [
    "races_analyzed",
    "horses_analyzed",
    "gold_high_iq",
    "race_results",
    "race_results_summary",
]
for table in tables:
    try:
        count = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count}")
    except:
        print(f"  {table}: ERROR or doesn't exist")

# Check pending races
print("\n⏳ PENDING RACES:")
pending = c.execute("""
    SELECT COUNT(*) FROM races_analyzed 
    WHERE race_id NOT IN (SELECT race_id FROM race_results_summary)
""").fetchone()[0]
print(f"  Analyzed but no results: {pending}")

# Check orphaned results
print("\n🚨 DATA INTEGRITY:")
orphaned = c.execute("""
    SELECT COUNT(*) FROM race_results_summary 
    WHERE race_id NOT IN (SELECT race_id FROM races_analyzed)
""").fetchone()[0]
print(f"  Results without analyzed race: {orphaned}")

if orphaned > 0:
    print("\n  Orphaned race_ids:")
    for row in c.execute("""
        SELECT race_id FROM race_results_summary 
        WHERE race_id NOT IN (SELECT race_id FROM races_analyzed)
    """).fetchall():
        print(f"    - {row[0]}")

# Calculate totals matching app display
completed = c.execute("SELECT COUNT(*) FROM race_results_summary").fetchone()[0]
pending_count = c.execute("""
    SELECT COUNT(*) FROM races_analyzed 
    WHERE race_id NOT IN (SELECT race_id FROM race_results_summary)
""").fetchone()[0]
total_shown = completed + pending_count

print("\n🖥️  APP DISPLAY LOGIC:")
print(
    f"  Total Analyzed: {total_shown} (= {completed} completed + {pending_count} pending)"
)
print(f"  With Results: {completed}")
print(f"  Pending Results: {pending_count}")

# Actual races_analyzed count
actual_analyzed = c.execute("SELECT COUNT(*) FROM races_analyzed").fetchone()[0]
print(f"\n✅ ACTUAL races_analyzed: {actual_analyzed}")
print(f"❓ Expected (based on app): {total_shown}")
print(f"📉 Difference: {total_shown - actual_analyzed} (should be 0)")

if total_shown != actual_analyzed:
    print(
        f"\n⚠️  MISMATCH DETECTED! The app shows {total_shown} but database has {actual_analyzed}"
    )
    if orphaned > 0:
        print(
            f"   → Likely cause: {orphaned} orphaned result(s) counted by app but not in races_analyzed"
        )

conn.close()

print("\n" + "=" * 60)
print("Fix: Delete orphaned results or re-analyze those races")
print("=" * 60)
