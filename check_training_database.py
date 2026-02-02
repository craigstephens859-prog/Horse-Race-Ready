import sqlite3

conn = sqlite3.connect('gold_high_iq.db')
cursor = conn.cursor()

# Get all races
cursor.execute('SELECT DISTINCT race_id FROM gold_high_iq ORDER BY race_id')
races = cursor.fetchall()

print(f"\n{'='*60}")
print(f"RACES IN TRAINING DATABASE: {len(races)} total")
print(f"{'='*60}\n")

for r in races:
    race_id = r[0]
    cursor.execute('SELECT COUNT(*) FROM gold_high_iq WHERE race_id = ?', (race_id,))
    horse_count = cursor.fetchone()[0]
    print(f"  {race_id:30s} - {horse_count} horses")

# Get date range
cursor.execute('''
    SELECT MIN(date), MAX(date), COUNT(DISTINCT race_id) 
    FROM gold_high_iq
''')
min_date, max_date, race_count = cursor.fetchone()

print(f"\n{'='*60}")
print(f"DATE RANGE: {min_date} to {max_date}")
print(f"TOTAL RACES: {race_count}")
print(f"{'='*60}\n")

conn.close()
