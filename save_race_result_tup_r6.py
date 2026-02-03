"""
Save TUP R6 (Feb 2, 2026) - Allowance $50k - MODEL FAILURE CASE
Winner: #5 Cactus League (7/1) - C-Group longshot
Failed Pick: #2 Ez Cowboy (94.8% probability, finished 4th)

This race is critical for model calibration - first failure after 4 perfect SA races.
Used to validate allowance race component weight adjustments.
"""

import sqlite3
import os

def save_tup_r6_allowance():
    """Save TUP R6 allowance race results to gold_high_iq database"""
    
    db_path = os.path.join(os.path.dirname(__file__), 'gold_high_iq.db')
    
    # Race metadata
    race_id = 'TUP_R6_20260202'
    track = 'Turf Paradise'
    race_num = 6
    race_date = '2026-02-02'
    distance = 6.0
    surface = 'Dirt'
    race_type = 'Allowance'
    purse = 50000
    
    # Actual finish order: 5-1-3-2-7
    horses_data = [
        {
            'program_number': 5,
            'horse_name': 'Cactus League',
            'actual_finish_position': 1,  # WINNER
            'predicted_finish_position': 6,  # Model predicted C-Group
            'prediction_error': 5,  # Missed by 5 positions
            'odds': 7.0,
            'prime_power': 109.1,
            'last_speed_rating': 66,
            'best_speed_at_distance': 69,
            'class_rating': 69,
            'running_style': 'E',
            'post_position': 5,
            'jockey': 'Alvarado Frank T',
            'trainer': 'Eikleberry Kevin (22%)',
            'days_since_last_race': 18,
            'career_starts': 3,
            'notes': 'WINNER - Hot trainer (22%, 4-0-0 L14), 2nd time Lasix 33% angle, C-Group rating missed by model'
        },
        {
            'program_number': 1,
            'horse_name': 'Saint Benji',
            'actual_finish_position': 2,
            'predicted_finish_position': 3,
            'prediction_error': 1,
            'odds': 3.0,
            'prime_power': 111.3,
            'last_speed_rating': 68,
            'best_speed_at_distance': 72,
            'class_rating': 72,
            'running_style': 'E/P',
            'post_position': 1,
            'jockey': 'Corbett Glenn W',
            'trainer': 'Evans Justin R (16%)',
            'days_since_last_race': 34,
            'career_starts': 4,
            'notes': 'Predicted 3rd, finished 2nd - close'
        },
        {
            'program_number': 3,
            'horse_name': 'Secret Insanity',
            'actual_finish_position': 3,
            'predicted_finish_position': 4,
            'prediction_error': 1,
            'odds': 2.0,
            'prime_power': 112.0,
            'last_speed_rating': 73,
            'best_speed_at_distance': 74,
            'class_rating': 74,
            'running_style': 'P',
            'post_position': 3,
            'jockey': 'Montalvo Carlos',
            'trainer': 'Eikleberry Kevin (22%)',
            'days_since_last_race': 18,
            'career_starts': 3,
            'notes': 'Smart Money horse (ML 9/2->2/1), predicted 4th, finished 3rd'
        },
        {
            'program_number': 2,
            'horse_name': 'Ez Cowboy',
            'actual_finish_position': 4,
            'predicted_finish_position': 1,  # MODEL PREDICTED WINNER
            'prediction_error': 3,
            'odds': 5.0,
            'prime_power': 114.6,
            'last_speed_rating': 68,
            'best_speed_at_distance': 75,
            'class_rating': 75,
            'running_style': 'E',
            'post_position': 2,
            'jockey': 'Americano Manuel',
            'trainer': 'Belvoir Vann (10%)',
            'days_since_last_race': 18,
            'career_starts': 8,
            'notes': 'FAILED PICK - Model gave 94.8% probability, finished 4th. High class (75) but low trainer %, no hot angles'
        },
        {
            'program_number': 7,
            'horse_name': 'Marking Broadway',
            'actual_finish_position': 5,
            'predicted_finish_position': 5,
            'prediction_error': 0,
            'odds': 3.5,
            'prime_power': 117.0,
            'last_speed_rating': 77,
            'best_speed_at_distance': 77,
            'class_rating': 77,
            'running_style': 'E/P',
            'post_position': 7,
            'jockey': 'Amador Silvio R',
            'trainer': 'Williams Dewey (0%)',
            'days_since_last_race': 145,
            'career_starts': 4,
            'notes': 'Predicted 5th correctly - long layoff hurt'
        },
    ]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n{'='*70}")
        print(f"Saving TUP R6 Allowance Race Results")
        print(f"{'='*70}")
        print(f"Race: {track} R{race_num} - {race_date}")
        print(f"Type: {race_type} ${purse:,}")
        print(f"Distance: {distance}f {surface}")
        print(f"\nActual Finish Order: 5-1-3-2-7")
        print(f"Model Predicted Order: 2-6-1-3-7 (MISSED WINNER!)")
        print(f"\n{'='*70}\n")
        
        for horse in horses_data:
            # Insert into database
            cursor.execute("""
                INSERT INTO gold_high_iq (
                    race_id, track, race_num, race_date, distance, surface, race_type,
                    horse_name, program_number, 
                    actual_finish_position, predicted_finish_position, prediction_error,
                    odds, prime_power, last_speed_rating, class_rating, best_speed_at_distance,
                    running_style, days_since_last_race, career_starts,
                    jockey, trainer, post_position, field_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id, track, race_num, race_date, distance, surface, race_type,
                horse['horse_name'], horse['program_number'],
                horse['actual_finish_position'], horse['predicted_finish_position'], horse['prediction_error'],
                horse['odds'], horse['prime_power'], 
                horse['last_speed_rating'], horse['class_rating'], horse['best_speed_at_distance'],
                horse['running_style'], horse['days_since_last_race'], horse['career_starts'],
                horse['jockey'], horse['trainer'], horse['post_position'], 9  # field_size
            ))
            
            # Print summary
            finish = horse['actual_finish_position']
            predicted = horse['predicted_finish_position']
            error = horse['prediction_error']
            
            status = "✅ CORRECT" if error == 0 else "❌ MISSED" if error >= 3 else "⚠️ CLOSE"
            
            print(f"#{horse['program_number']} {horse['horse_name']:<20} "
                  f"Finish: {finish:>2} | Predicted: {predicted:>2} | Error: {error:>2} {status}")
            
            if 'notes' in horse:
                print(f"   → {horse['notes']}")
        
        conn.commit()
        
        print(f"\n{'='*70}")
        print("✅ TUP R6 saved successfully to gold_high_iq.db")
        print(f"{'='*70}")
        
        # Calculate accuracy
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_finish_position = actual_finish_position THEN 1 ELSE 0 END) as exact_correct,
                SUM(CASE WHEN ABS(predicted_finish_position - actual_finish_position) <= 1 THEN 1 ELSE 0 END) as within_1
            FROM gold_high_iq
            WHERE race_id = ?
        """, (race_id,))
        
        total, exact, within_1 = cursor.fetchone()
        
        print(f"\nTUP R6 Accuracy:")
        print(f"  Exact Predictions: {exact}/{total} ({exact/total*100:.1f}%)")
        print(f"  Within 1 Position: {within_1}/{total} ({within_1/total*100:.1f}%)")
        print(f"  Winner Predicted: ❌ NO (missed winner #5)")
        
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving TUP R6: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    save_tup_r6_allowance()
