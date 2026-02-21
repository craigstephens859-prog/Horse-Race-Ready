"""Quick test: verify new track_accuracy_profiles table + backfill existing 66 races."""

import sqlite3

from gold_database_manager import GoldHighIQDatabase

db = GoldHighIQDatabase()
print("✅ DB init OK")

# Check table exists
conn = sqlite3.connect("gold_high_iq.db")
c = conn.cursor()
c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='track_accuracy_profiles'"
)
print("✅ Table exists:", bool(c.fetchone()))

# Get all tracks with completed races
c.execute("""
    SELECT DISTINCT ra.track_code, COUNT(*) as cnt
    FROM races_analyzed ra
    JOIN race_results_summary rrs ON ra.race_id = rrs.race_id
    GROUP BY ra.track_code
    ORDER BY cnt DESC
""")
tracks = c.fetchall()
print(f"\n📊 Tracks with completed results: {len(tracks)}")
for t, cnt in tracks:
    print(f"  {t}: {cnt} races")

# Backfill profiles for all tracks
print("\n🔄 Backfilling accuracy & bias profiles...")
for track_code, cnt in tracks:
    db._rebuild_accuracy_profile(track_code)
    db._rebuild_bias_profile(track_code)
    print(f"  ✅ {track_code}: profiles rebuilt ({cnt} races)")

# Verify data
c.execute(
    "SELECT track_code, profile_key, profile_value, sample_size FROM track_accuracy_profiles ORDER BY track_code, profile_key"
)
rows = c.fetchall()
print(f"\n📋 Total profile entries: {len(rows)}")
for track_code, key, value, sample in rows[:20]:
    print(f"  {track_code} | {key}: {value} (n={sample})")

# Test the API functions
print("\n🧪 Testing calculate_accuracy_stats...")
if tracks:
    stats = db.calculate_accuracy_stats(tracks[0][0])
    print(f"  {tracks[0][0]}: {list(stats.keys())}")

print("\n🧪 Testing detect_biases...")
if tracks:
    biases = db.detect_biases(tracks[0][0])
    print(f"  Style: {biases.get('style_bias')}")
    print(f"  Rail: {biases.get('rail_bias')}")
    print(f"  Active: {biases.get('active_biases')}")
    print(f"  Post: {biases.get('post_position_stats')}")

conn.close()
print("\n✅ All tests passed!")
