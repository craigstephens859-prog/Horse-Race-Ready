"""
Fix database consistency by removing orphaned race result
"""
import sqlite3

DB_FILE = "gold_high_iq.db"

print("🔧 DATABASE CONSISTENCY FIX")
print("=" * 60)

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Find orphaned results
print("\n🔍 Finding orphaned results...")
orphaned = c.execute("""
    SELECT race_id FROM race_results_summary 
    WHERE race_id NOT IN (SELECT race_id FROM races_analyzed)
""").fetchall()

if not orphaned:
    print("✅ No orphaned results found! Database is consistent.")
    conn.close()
    exit(0)

print(f"Found {len(orphaned)} orphaned result(s):")
for row in orphaned:
    race_id = row[0]
    print(f"  - {race_id}")

# Delete orphaned results
print(f"\n🗑️  Deleting orphaned results...")
for row in orphaned:
    race_id = row[0]
    # Delete from race_results_summary
    c.execute("DELETE FROM race_results_summary WHERE race_id = ?", (race_id,))
    print(f"  ✓ Deleted {race_id} from race_results_summary")

conn.commit()
print("\n✅ Database consistency fixed!")

# Verify
print("\n📊 AFTER FIX:")
analyzed = c.execute("SELECT COUNT(*) FROM races_analyzed").fetchone()[0]
completed = c.execute("SELECT COUNT(*) FROM race_results_summary").fetchone()[0]
pending = c.execute("""
    SELECT COUNT(*) FROM races_analyzed 
    WHERE race_id NOT IN (SELECT race_id FROM race_results_summary)
""").fetchone()[0]

print(f"  races_analyzed: {analyzed}")
print(f"  race_results_summary: {completed}")
print(f"  Pending races: {pending}")
print(f"  Total shown in app: {completed + pending}")
print(f"\n✅ Match: {analyzed == completed + pending}")

conn.close()
print("\n" + "=" * 60)
print("Done! Restart Streamlit app to see updated counts.")
print("=" * 60)
