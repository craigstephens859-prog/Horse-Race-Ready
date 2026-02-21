import sqlite3

conn = sqlite3.connect("gold_high_iq.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Find TUP R6 races
cur.execute("""SELECT race_id, track_code, race_number, race_date, surface, distance, track_condition, field_size
    FROM races_analyzed 
    WHERE track_code LIKE '%TUP%' AND race_number=6
    ORDER BY race_date DESC LIMIT 5""")
rows = cur.fetchall()
for r in rows:
    print(dict(r))

print("---")

# Check horses_analyzed schema
cur.execute("PRAGMA table_info(horses_analyzed)")
cols = cur.fetchall()
print("horses_analyzed columns:")
for c in cols:
    print(f"  {c['name']}")

print("---")

# Get horses for the most recent TUP R6
if rows:
    rid = rows[0]["race_id"]
    print(f"\nHorses for race_id: {rid}")
    cur.execute(
        """SELECT horse_name, final_rating, predicted_rank, actual_finish_position,
        jockey_name, morning_line_odds
        FROM horses_analyzed 
        WHERE race_id=?
        ORDER BY predicted_rank""",
        (rid,),
    )
    horses = cur.fetchall()
    for h in horses:
        print(dict(h))

conn.close()
