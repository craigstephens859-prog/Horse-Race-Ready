"""Detailed race data check with correct schema."""
import sqlite3

conn = sqlite3.connect("gold_high_iq.db")
c = conn.cursor()

# Get races_analyzed schema
print("=== RACES_ANALYZED SCHEMA ===")
c.execute("PRAGMA table_info(races_analyzed)")
for col in c.fetchall():
    print(f"  {col[1]:25s} {col[2]}")

print("\n=== ALL RACES ANALYZED ===")
c.execute("SELECT * FROM races_analyzed ORDER BY rowid DESC")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
print(f"Columns: {col_names}")
print(f"Total: {len(rows)} races")
for r in rows:
    print(f"  {dict(zip(col_names, r))}")

# Get gold_high_iq schema
print("\n=== GOLD_HIGH_IQ SCHEMA ===")
c.execute("PRAGMA table_info(gold_high_iq)")
for col in c.fetchall():
    print(f"  {col[1]:30s} {col[2]}")

print("\n=== ALL GOLD_HIGH_IQ ENTRIES ===")
c.execute("SELECT * FROM gold_high_iq ORDER BY rowid DESC")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
print(f"Columns: {col_names}")
print(f"Total: {len(rows)} rows")
for r in rows:
    d = dict(zip(col_names, r))
    # Print compact version
    print(f"\n  Race: {d.get('race_id','?')}")
    print(f"    Horse: {d.get('horse_name','?')}")
    print(f"    Predicted Prob: {d.get('predicted_probability','?')}")
    print(f"    Actual Finish: {d.get('actual_finish_position','?')}")
    print(f"    Result Timestamp: {d.get('result_entered_timestamp','?')}")

# Check horses_analyzed too
print("\n=== HORSES_ANALYZED SCHEMA ===")
c.execute("PRAGMA table_info(horses_analyzed)")
for col in c.fetchall():
    print(f"  {col[1]:30s} {col[2]}")

print("\n=== ALL HORSES_ANALYZED (last 20) ===")
c.execute("SELECT race_id, horse_name, program_number, predicted_rank, predicted_probability FROM horses_analyzed ORDER BY rowid DESC LIMIT 20")
rows = c.fetchall()
print(f"Total shown: {len(rows)}")
for r in rows:
    prob = f"{r[4]:.1%}" if r[4] else "N/A"
    print(f"  {r[0]} | #{r[2]:2} {r[1]:25s} | Rank: {r[3]} | Prob: {prob}")

# Check race_results
print("\n=== RACE_RESULTS ===")
c.execute("SELECT * FROM race_results ORDER BY rowid DESC LIMIT 20")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
print(f"Columns: {col_names}")
for r in rows:
    print(f"  {dict(zip(col_names, r))}")

# Check race_results_summary
print("\n=== RACE_RESULTS_SUMMARY ===")
try:
    c.execute("SELECT * FROM race_results_summary ORDER BY rowid DESC LIMIT 10")
    rows = c.fetchall()
    col_names = [d[0] for d in c.description]
    print(f"Columns: {col_names}")
    for r in rows:
        print(f"  {dict(zip(col_names, r))}")
except Exception as e:
    print(f"  Error: {e}")

conn.close()
