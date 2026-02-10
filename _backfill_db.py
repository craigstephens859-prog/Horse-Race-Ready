"""
Backfill races_analyzed, horses_analyzed, and race_results_summary
from existing gold_high_iq data so the community banner and saved count
show the correct numbers on Render.
"""
import sqlite3
from datetime import datetime

conn = sqlite3.connect('gold_high_iq.db')
c = conn.cursor()

# Get distinct races from gold_high_iq
c.execute("""
    SELECT race_id, track, race_num, race_date, race_type, surface, distance, field_size,
           MIN(timestamp) as first_ts
    FROM gold_high_iq
    GROUP BY race_id
""")
races = c.fetchall()
print(f"Found {len(races)} races in gold_high_iq to backfill")

for race in races:
    race_id, track, race_num, race_date, race_type, surface, distance, field_size, ts = race
    if not ts:
        ts = datetime.now().isoformat()
    
    # 1. Backfill races_analyzed
    c.execute("SELECT COUNT(*) FROM races_analyzed WHERE race_id = ?", (race_id,))
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO races_analyzed 
            (race_id, track_code, race_date, race_number, race_type, surface, distance, field_size, analyzed_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (race_id, track, race_date, race_num, race_type, surface, str(distance), field_size, ts))
        print(f"  ✅ races_analyzed: {race_id}")
    
    # 2. Backfill horses_analyzed
    c.execute("""
        SELECT horse_name, program_number, post_position, odds, jockey, trainer,
               running_style, prime_power, last_speed_rating, class_rating,
               days_since_last_race, career_starts, predicted_finish_position
        FROM gold_high_iq
        WHERE race_id = ?
        ORDER BY predicted_finish_position
    """, (race_id,))
    horses = c.fetchall()
    
    for idx, h in enumerate(horses):
        name, pgm, post, odds, jockey, trainer, style, pp_rating, beyer, cls_rating, days, starts, pred_rank = h
        horse_id = f"{race_id}_{pgm}"
        c.execute("SELECT COUNT(*) FROM horses_analyzed WHERE horse_id = ?", (horse_id,))
        if c.fetchone()[0] == 0:
            pred_prob = max(0.01, 1.0 / (pred_rank + 0.5)) if pred_rank else 0.05
            c.execute("""
                INSERT INTO horses_analyzed
                (horse_id, race_id, program_number, horse_name, post_position,
                 morning_line_odds, jockey, trainer, running_style, prime_power,
                 last_beyer, class_rating, days_since_last, starts_lifetime,
                 predicted_probability, predicted_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (horse_id, race_id, pgm, name, post, odds, jockey, trainer,
                  style, pp_rating, beyer, cls_rating, days, starts, pred_prob, pred_rank))
    print(f"  ✅ horses_analyzed: {len(horses)} horses for {race_id}")
    
    # 3. Backfill race_results_summary (we have actual finish positions)
    c.execute("SELECT COUNT(*) FROM race_results_summary WHERE race_id = ?", (race_id,))
    if c.fetchone()[0] == 0:
        # Get actual finishers by position
        c.execute("""
            SELECT horse_name, actual_finish_position, predicted_finish_position
            FROM gold_high_iq
            WHERE race_id = ?
            ORDER BY actual_finish_position
        """, (race_id,))
        finishers = c.fetchall()
        
        # Build finish order
        by_pos = {int(f[1]): f[0] for f in finishers if f[1] is not None and f[1] > 0}
        winner = by_pos.get(1, '')
        second = by_pos.get(2, '')
        third = by_pos.get(3, '')
        fourth = by_pos.get(4, '')
        fifth = by_pos.get(5, '')
        
        # Check prediction accuracy
        pred_by_rank = {int(f[2]): f[0] for f in finishers if f[2] is not None}
        predicted_winner = pred_by_rank.get(1, '')
        top1_correct = (predicted_winner == winner) if winner else False
        
        # Count how many of predicted top 3 actually finished top 3
        actual_top3 = {by_pos.get(i, '') for i in [1, 2, 3]} - {''}
        pred_top3 = {pred_by_rank.get(i, '') for i in [1, 2, 3]} - {''}
        top3_correct = len(actual_top3 & pred_top3)
        
        actual_top5 = {by_pos.get(i, '') for i in [1, 2, 3, 4, 5]} - {''}
        pred_top5 = {pred_by_rank.get(i, '') for i in [1, 2, 3, 4, 5]} - {''}
        top5_correct = len(actual_top5 & pred_top5)
        
        c.execute("""
            INSERT INTO race_results_summary
            (race_id, winner_name, second_name, third_name, fourth_name, fifth_name,
             top1_predicted_correctly, top3_predicted_correctly, top5_predicted_correctly,
             results_complete_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (race_id, winner, second, third, fourth, fifth,
              top1_correct, top3_correct, top5_correct,
              ts))
        print(f"  ✅ race_results_summary: {race_id} (winner={winner}, top1_correct={top1_correct})")

conn.commit()

# Verify
print("\n--- Final counts ---")
for table in ['races_analyzed', 'horses_analyzed', 'race_results_summary', 'gold_high_iq']:
    c.execute(f"SELECT COUNT(*) FROM [{table}]")
    print(f"  {table}: {c.fetchone()[0]} rows")

conn.close()
print("\n✅ Backfill complete!")
