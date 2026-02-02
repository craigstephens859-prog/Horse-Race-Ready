"""
Save Santa Anita Race 8 Results to Database for ML Training
============================================================
Race: SA R8, February 1, 2026
Distance: 6F Dirt Sprint
Type: Claiming $20,000
Actual Finish: 13-12-8-5-9

Key Insight: Prime Power predicted PERFECTLY (top 3 PP = top 3 finishers)
This race validated the 92/8 PP weight for experienced dirt sprints
Winner #13 Rizzleberry Rose: PP 125.3 (3rd highest)
"""

import sqlite3
import os

# Race metadata
RACE_DATA = {
    'race_id': 'SA_R8_20260201',
    'track': 'Santa Anita',
    'race_num': 8,
    'date': '2026-02-01',
    'distance': 6.0,  # furlongs
    'surface': 'Dirt',
    'race_type': 'CLM 20000',
    'purse': 22000,
    'field_size': 13,
    'conditions': 'F&M 3yo+ Claiming $20k'
}

# Complete field with all 13 horses
HORSES = [
    {
        'horse_num': 13,
        'horse_name': 'Rizzleberry Rose',
        'finish_position': 1,  # WINNER
        'final_odds': 4.5,
        'speed_last': 77,
        'speed_best': 83,
        'prime_power': 125.3,  # 3rd highest PP - Won!
        'class_rating': 113.5,
        'e1_pace': 92,
        'e2_late': 74,
        'running_style': 'E',
        'form_cycle': 0.85,
        'jockey_win_pct': 0.22,  # Elite jockey
        'trainer_win_pct': 0.18,  # Elite trainer
        'days_since_last': 14,
        'workouts_count': 3,
        'beaten_lengths': 0.0
    },
    {
        'horse_num': 12,
        'horse_name': 'Miss Practical',
        'finish_position': 2,
        'final_odds': 3.0,  # Favorite
        'speed_last': 82,
        'speed_best': 89,
        'prime_power': 127.5,  # HIGHEST PP - Finished 2nd
        'class_rating': 114.8,
        'e1_pace': 85,
        'e2_late': 88,
        'running_style': 'S',
        'form_cycle': 0.92,
        'jockey_win_pct': 0.19,
        'trainer_win_pct': 0.17,
        'days_since_last': 21,
        'workouts_count': 4,
        'beaten_lengths': 0.75
    },
    {
        'horse_num': 8,
        'horse_name': 'Stay in Line',
        'finish_position': 3,
        'final_odds': 6.0,
        'speed_last': 73,
        'speed_best': 93,
        'prime_power': 125.4,  # 2nd highest PP - Finished 3rd
        'class_rating': 113.9,
        'e1_pace': 87,
        'e2_late': 75,
        'running_style': 'E',
        'form_cycle': 0.78,
        'jockey_win_pct': 0.16,
        'trainer_win_pct': 0.14,
        'days_since_last': 17,
        'workouts_count': 3,
        'beaten_lengths': 1.5
    },
    {
        'horse_num': 5,
        'horse_name': 'Clubhouse Bride',
        'finish_position': 4,
        'final_odds': 20.0,
        'speed_last': 77,
        'speed_best': 88,
        'prime_power': 122.3,
        'class_rating': 113.6,
        'e1_pace': 88,
        'e2_late': 79,
        'running_style': 'E/P',
        'form_cycle': 0.65,
        'jockey_win_pct': 0.12,
        'trainer_win_pct': 0.11,
        'days_since_last': 24,
        'workouts_count': 2,
        'beaten_lengths': 3.0
    },
    {
        'horse_num': 9,
        'horse_name': 'Maniae',
        'finish_position': 5,
        'final_odds': 30.0,
        'speed_last': 65,
        'speed_best': 83,
        'prime_power': 114.6,
        'class_rating': 111.3,
        'e1_pace': 87,
        'e2_late': 70,
        'running_style': 'P',
        'form_cycle': 0.42,
        'jockey_win_pct': 0.09,
        'trainer_win_pct': 0.08,
        'days_since_last': 35,
        'workouts_count': 2,
        'beaten_lengths': 5.0
    },
    # Remaining horses (did not finish in top 5)
    {
        'horse_num': 1,
        'horse_name': 'Clarina',
        'finish_position': 99,  # DNF top 5
        'final_odds': 8.0,
        'speed_last': 91,
        'speed_best': 91,
        'prime_power': 117.7,
        'class_rating': 113.3,
        'e1_pace': 99,
        'e2_late': 87,
        'running_style': 'E/P',
        'form_cycle': 0.88,
        'jockey_win_pct': 0.15,
        'trainer_win_pct': 0.13,
        'days_since_last': 20,
        'workouts_count': 3,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 2,
        'horse_name': 'Timekeeper\'s Charm',
        'finish_position': 99,
        'final_odds': 6.0,
        'speed_last': 78,
        'speed_best': 89,
        'prime_power': 124.4,
        'class_rating': 114.4,
        'e1_pace': 95,
        'e2_late': 73,
        'running_style': 'E/P',
        'form_cycle': 0.70,
        'jockey_win_pct': 0.14,
        'trainer_win_pct': 0.12,
        'days_since_last': 18,
        'workouts_count': 2,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 3,
        'horse_name': 'Lavender Love',
        'finish_position': 99,
        'final_odds': 8.0,
        'speed_last': 79,
        'speed_best': 82,
        'prime_power': 123.5,
        'class_rating': 114.6,
        'e1_pace': 78,
        'e2_late': 85,
        'running_style': 'E',
        'form_cycle': 0.75,
        'jockey_win_pct': 0.13,
        'trainer_win_pct': 0.11,
        'days_since_last': 22,
        'workouts_count': 3,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 4,
        'horse_name': 'Fibonaccis Ride',
        'finish_position': 99,
        'final_odds': 15.0,
        'speed_last': 77,
        'speed_best': 84,
        'prime_power': 120.4,
        'class_rating': 112.5,
        'e1_pace': 93,
        'e2_late': 71,
        'running_style': 'E',
        'form_cycle': 0.60,
        'jockey_win_pct': 0.10,
        'trainer_win_pct': 0.09,
        'days_since_last': 28,
        'workouts_count': 2,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 6,
        'horse_name': 'Clubhouse Cutie',
        'finish_position': 99,
        'final_odds': 20.0,
        'speed_last': 81,
        'speed_best': 81,
        'prime_power': 121.0,
        'class_rating': 112.5,
        'e1_pace': 94,
        'e2_late': 80,
        'running_style': 'E',
        'form_cycle': 0.68,
        'jockey_win_pct': 0.11,
        'trainer_win_pct': 0.10,
        'days_since_last': 25,
        'workouts_count': 2,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 7,
        'horse_name': 'Ryan\'s Girl',
        'finish_position': 99,
        'final_odds': 20.0,
        'speed_last': 80,
        'speed_best': 88,
        'prime_power': 118.1,
        'class_rating': 112.3,
        'e1_pace': 88,
        'e2_late': 73,
        'running_style': 'E/P',
        'form_cycle': 0.72,
        'jockey_win_pct': 0.12,
        'trainer_win_pct': 0.10,
        'days_since_last': 19,
        'workouts_count': 3,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 10,
        'horse_name': 'Petite Treat',
        'finish_position': 99,
        'final_odds': 20.0,
        'speed_last': 76,
        'speed_best': 82,
        'prime_power': 116.6,
        'class_rating': 112.5,
        'e1_pace': 88,
        'e2_late': 79,
        'running_style': 'P',
        'form_cycle': 0.58,
        'jockey_win_pct': 0.09,
        'trainer_win_pct': 0.08,
        'days_since_last': 31,
        'workouts_count': 2,
        'beaten_lengths': 99.0
    },
    {
        'horse_num': 11,
        'horse_name': 'Sexy Blue',
        'finish_position': 99,
        'final_odds': 20.0,
        'speed_last': 72,
        'speed_best': 87,
        'prime_power': 119.0,
        'class_rating': 112.8,
        'e1_pace': 84,
        'e2_late': 77,
        'running_style': 'E/P',
        'form_cycle': 0.55,
        'jockey_win_pct': 0.10,
        'trainer_win_pct': 0.09,
        'days_since_last': 26,
        'workouts_count': 2,
        'beaten_lengths': 99.0
    }
]


def save_to_database():
    """Save SA R8 results to gold_high_iq.db"""
    db_path = os.path.join(os.path.dirname(__file__), 'gold_high_iq.db')
    
    print("=" * 80)
    print("🏇 SAVING SANTA ANITA RACE 8 RESULTS TO DATABASE")
    print("=" * 80)
    print(f"Race ID: {RACE_DATA['race_id']}")
    print(f"Track: {RACE_DATA['track']} R{RACE_DATA['race_num']}")
    print(f"Date: {RACE_DATA['date']}")
    print(f"Distance: {RACE_DATA['distance']}F {RACE_DATA['surface']}")
    print(f"Type: {RACE_DATA['race_type']}")
    print(f"Field: {RACE_DATA['field_size']} horses")
    print(f"Actual Finish: 13-12-8-5-9")
    print(f"Key: Top 3 PP horses = Top 3 finishers (Perfect PP correlation)")
    print()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if race already exists
        cursor.execute('SELECT COUNT(*) FROM gold_high_iq WHERE race_id = ?', 
                      (RACE_DATA['race_id'],))
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️  Race {RACE_DATA['race_id']} already exists ({existing_count} horses)")
            print("    Skipping to avoid duplicates.")
            return
        
        # Insert each horse
        saved_count = 0
        for horse in HORSES:
            cursor.execute('''
                INSERT INTO gold_high_iq (
                    race_id, track, race_num, race_date, distance, surface, race_type,
                    horse_name, program_number, actual_finish_position, odds,
                    last_speed_rating, best_speed_at_distance, prime_power, class_rating,
                    running_style, days_since_last_race, field_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                RACE_DATA['race_id'],
                RACE_DATA['track'],
                RACE_DATA['race_num'],
                RACE_DATA['date'],
                RACE_DATA['distance'],
                RACE_DATA['surface'],
                RACE_DATA['race_type'],
                horse['horse_name'],
                horse['horse_num'],
                horse['finish_position'],
                horse['final_odds'],
                horse['speed_last'],
                horse['speed_best'],
                horse['prime_power'],
                horse['class_rating'],
                horse['running_style'],
                horse['days_since_last'],
                RACE_DATA['field_size']
            ))
            saved_count += 1
            
            if horse['finish_position'] <= 5:
                print(f"✓ Saved #{horse['horse_num']:2d} {horse['horse_name']:20s} - Finished {horse['finish_position']} (PP:{horse['prime_power']:.1f})")
            else:
                print(f"  Saved #{horse['horse_num']:2d} {horse['horse_name']:20s} - DNF top 5")
        
        conn.commit()
        print()
        print(f"✅ SUCCESS! Saved {saved_count} horses to database")
        print()
        
        # Verify total count
        cursor.execute('SELECT COUNT(DISTINCT race_id) FROM gold_high_iq')
        total_races = cursor.fetchone()[0]
        print(f"📊 Database now contains: {total_races} races")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == '__main__':
    save_to_database()
