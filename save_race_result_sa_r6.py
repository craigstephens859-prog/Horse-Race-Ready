"""
Save Santa Anita Race 6 Results to Database for ML Training
============================================================
Race: SA R6, February 1, 2026
Distance: 6F Dirt Sprint
Type: Claiming $16,000
Actual Finish: 4-7-8-3-2

Key Insight: Class dropper scenario - winner #4 Windribbon had +0.45 class advantage
This race validated the need for class weight adjustments (3.0× multiplier)
"""

import sqlite3
import os

# Race metadata
RACE_DATA = {
    'race_id': 'SA_R6_20260201',
    'track': 'Santa Anita',
    'race_num': 6,
    'date': '2026-02-01',
    'distance': 6.0,  # furlongs
    'surface': 'Dirt',
    'race_type': 'CLM 16000',
    'purse': 18000,
    'field_size': 5,
    'conditions': '3yo+ Claiming $16k'
}

# Horse data - Top 5 finishers with complete features
HORSES = [
    {
        'horse_num': 4,
        'horse_name': 'Windribbon',
        'finish_position': 1,  # WINNER
        'final_odds': 3.2,
        'speed_last': 76,
        'speed_best': 76,
        'prime_power': 121.0,  # Estimated based on speed/class
        'class_rating': 113.5,  # +0.45 class advantage (key factor)
        'e1_pace': 88,
        'e2_late': 80,
        'running_style': 'E/P',
        'form_cycle': 1.00,  # Peak form
        'jockey_win_pct': 0.15,
        'trainer_win_pct': 0.12,
        'days_since_last': 14,
        'workouts_count': 2,
        'beaten_lengths': 0.0
    },
    {
        'horse_num': 7,
        'horse_name': 'Big Cheeseola',
        'finish_position': 2,
        'final_odds': 2.1,  # Favorite
        'speed_last': 67,
        'speed_best': 67,
        'prime_power': 123.0,  # Highest PP but lost
        'class_rating': 114.5,  # Highest class (+1.00)
        'e1_pace': 82,
        'e2_late': 75,
        'running_style': 'P',
        'form_cycle': 0.65,
        'jockey_win_pct': 0.18,
        'trainer_win_pct': 0.16,
        'days_since_last': 21,
        'workouts_count': 3,
        'beaten_lengths': 1.5
    },
    {
        'horse_num': 8,
        'horse_name': 'Poise and Prada',
        'finish_position': 3,
        'final_odds': 8.5,
        'speed_last': 65,
        'speed_best': 65,
        'prime_power': 118.0,
        'class_rating': 111.5,  # Class drop (-0.75)
        'e1_pace': 79,
        'e2_late': 72,
        'running_style': 'S',
        'form_cycle': 0.48,
        'jockey_win_pct': 0.10,
        'trainer_win_pct': 0.09,
        'days_since_last': 28,
        'workouts_count': 2,
        'beaten_lengths': 3.0
    },
    {
        'horse_num': 3,
        'horse_name': 'Smarty Nose',
        'finish_position': 4,
        'final_odds': 4.8,
        'speed_last': 86,
        'speed_best': 86,
        'prime_power': 122.0,  # High PP
        'class_rating': 112.5,  # Class drop (-0.30)
        'e1_pace': 90,
        'e2_late': 78,
        'running_style': 'E',
        'form_cycle': 0.96,  # Good form
        'jockey_win_pct': 0.14,
        'trainer_win_pct': 0.13,
        'days_since_last': 17,
        'workouts_count': 3,
        'beaten_lengths': 4.5
    },
    {
        'horse_num': 2,
        'horse_name': 'Elegant Life',
        'finish_position': 5,
        'final_odds': 12.0,
        'speed_last': 69,
        'speed_best': 69,
        'prime_power': 120.0,
        'class_rating': 112.5,  # Class drop (-0.30)
        'e1_pace': 85,
        'e2_late': 76,
        'running_style': 'E/P',
        'form_cycle': 0.41,
        'jockey_win_pct': 0.08,
        'trainer_win_pct': 0.07,
        'days_since_last': 35,
        'workouts_count': 1,
        'beaten_lengths': 6.0
    }
]


def save_to_database():
    """Save SA R6 results to gold_high_iq.db"""
    db_path = os.path.join(os.path.dirname(__file__), 'gold_high_iq.db')
    
    print("=" * 80)
    print("🏇 SAVING SANTA ANITA RACE 6 RESULTS TO DATABASE")
    print("=" * 80)
    print(f"Race ID: {RACE_DATA['race_id']}")
    print(f"Track: {RACE_DATA['track']} R{RACE_DATA['race_num']}")
    print(f"Date: {RACE_DATA['date']}")
    print(f"Distance: {RACE_DATA['distance']}F {RACE_DATA['surface']}")
    print(f"Type: {RACE_DATA['race_type']}")
    print(f"Actual Finish: 4-7-8-3-2")
    print(f"Key: Class dropper scenario (winner had +0.45 class advantage)")
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
            print(f"✓ Saved #{horse['horse_num']} {horse['horse_name']:20s} - Finished {horse['finish_position']}")
        
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
