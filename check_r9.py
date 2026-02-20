import sqlite3

conn = sqlite3.connect("gold_high_iq.db")
c = conn.cursor()

# Check R9 race data
c.execute("SELECT * FROM races_analyzed WHERE race_id LIKE '%Oaklawn%R9%'")
rows = c.fetchall()
print("Race entry:")
for r in rows:
    print(r)

# Check if results exist
c.execute("SELECT * FROM race_results_summary WHERE race_id LIKE '%Oaklawn%R9%'")
results = c.fetchall()
print("\nResults entry:", results if results else "PENDING")

# All tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("\nAll tables:", [t[0] for t in c.fetchall()])

# Check pending races (no results yet)
c.execute("""
    SELECT ra.race_id FROM races_analyzed ra 
    LEFT JOIN race_results_summary rrs ON ra.race_id = rrs.race_id 
    WHERE rrs.race_id IS NULL
""")
pending = c.fetchall()
print("\nPending races (no results):")
for p in pending:
    print(f"  {p[0]}")

conn.close()
