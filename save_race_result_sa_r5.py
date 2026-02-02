"""
Save Santa Anita Race 5 Results to Database for ML Training
============================================================
Race: SA R5, February 1, 2026
Distance: 6½F Turf
Type: Maiden Special Weight $70k
Actual Finish: 3-6-7-8-1

Key Insight: TURF MAIDEN race - Highest PP (#4 Kizazi 116.9) didn't finish top 5
Winner #3 had 2nd highest PP (116.3), but 2nd/3rd were first-timers with no PP
This validates 0/100 weight for turf (components > Prime Power)
"""

import sqlite3
import os

# Race metadata
RACE_DATA = {
    'race_id': 'SA_R5_20260201',
    'track': 'Santa Anita',
    'race_num': 5,
    'date': '2026-02-01',
    'distance': 6.5,  # furlongs (6½F)
    'surface': 'Turf',
    'race_type': 'MSW 70000',
    'purse': 70000,
    'field_size': 8,
    'conditions': '3yo Fillies Maiden Special Weight'
}

# Complete field with all 8 horses
HORSES = [
    {
        'horse_num': 3,
        'horse_name': 'Surfin\' U. S. A.',
        'finish_position': 1,  # WINNER
        'final_odds': 8.0,
        'speed_last': 76,
        'speed_best': 81,
        'prime_power': 116.3,  # 2nd highest PP
        'class_rating': 111.7,
        'running_style': 'P',
        'days_since_last': 21,
        'post_position': 3,
        'ml_odds': 8.0
    },
    {
        'horse_num': 6,
        'horse_name': 'Fancy Lady',
        'finish_position': 2,
        'final_odds': 6.0,
        'speed_last': 0,  # First-time starter
        'speed_best': 0,
        'prime_power': 0.0,  # No PP (first-timer)
        'class_rating': 0.0,
        'running_style': 'NA',
        'days_since_last': 999,  # First start
        'post_position': 6,
        'ml_odds': 6.0
    },
    {
        'horse_num': 7,
        'horse_name': 'Acoustic Kitty',
        'finish_position': 3,
        'final_odds': 6.0,
        'speed_last': 0,  # First-time starter
        'speed_best': 0,
        'prime_power': 0.0,  # No PP (first-timer)
        'class_rating': 0.0,
        'running_style': 'NA',
        'days_since_last': 999,  # First start
        'post_position': 7,
        'ml_odds': 6.0
    },
    {
        'horse_num': 8,
        'horse_name': 'Silkie Sevei',
        'finish_position': 4,
        'final_odds': 2.5,  # Favorite
        'speed_last': 0,  # No US speed rating
        'speed_best': 0,
        'prime_power': 0.0,  # No PP listed
        'class_rating': 113.1,
        'running_style': 'NA',
        'days_since_last': 148,  # 148 days since last
        'post_position': 8,
        'ml_odds': 2.5
    },
    {
        'horse_num': 1,
        'horse_name': 'Red Cherry',
        'finish_position': 5,
        'final_odds': 4.0,
        'speed_last': 72,
        'speed_best': 72,
        'prime_power': 115.5,  # 3rd highest PP
        'class_rating': 111.5,
        'running_style': 'E',
        'days_since_last': 24,
        'post_position': 1,
        'ml_odds': 4.0
    },
    # Remaining horses (did not finish in top 5)
    {
        'horse_num': 4,
        'horse_name': 'Kizazi',
        'finish_position': 99,  # DNF top 5
        'final_odds': 5.0,
        'speed_last': 76,
        'speed_best': 78,
        'prime_power': 116.9,  # HIGHEST PP - didn't finish top 5!
        'class_rating': 112.3,
        'running_style': 'E',
        'days_since_last': 24,
        'post_position': 4,
        'ml_odds': 5.0
    },
    {
        'horse_num': 2,
        'horse_name': 'Lady Detective',
        'finish_position': 99,
        'final_odds': 20.0,
        'speed_last': 61,
        'speed_best': 61,
        'prime_power': 106.6,
        'class_rating': 108.9,
        'running_style': 'NA',
        'days_since_last': 24,
        'post_position': 2,
        'ml_odds': 20.0
    },
    {
        'horse_num': 5,
        'horse_name': 'Not With a Fox',
        'finish_position': 99,
        'final_odds': 4.5,
        'speed_last': 0,  # First-time starter
        'speed_best': 0,
        'prime_power': 0.0,
        'class_rating': 0.0,
        'running_style': 'NA',
        'days_since_last': 999,
        'post_position': 5,
        'ml_odds': 4.5
    }
]


def save_to_database():
    """Save SA R5 results to gold_high_iq.db"""
    db_path = os.path.join(os.path.dirname(__file__), 'gold_high_iq.db')
    
    print("=" * 80)
    print("🏇 SAVING SANTA ANITA RACE 5 RESULTS TO DATABASE")
    print("=" * 80)
    print(f"Race ID: {RACE_DATA['race_id']}")
    print(f"Track: {RACE_DATA['track']} R{RACE_DATA['race_num']}")
    print(f"Date: {RACE_DATA['date']}")
    print(f"Distance: {RACE_DATA['distance']}F {RACE_DATA['surface']}")
    print(f"Type: {RACE_DATA['race_type']}")
    print(f"Field: {RACE_DATA['field_size']} horses (3 first-timers)")
    print(f"Actual Finish: 3-6-7-8-1")
    print(f"Key: TURF race - Highest PP (#4: 116.9) didn't finish top 5")
    print(f"     Validates 0/100 turf weighting (components > Prime Power)")
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
                    running_style, days_since_last_race, field_size, post_position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                RACE_DATA['field_size'],
                horse['post_position']
            ))
            saved_count += 1
            
            if horse['finish_position'] <= 5:
                pp_str = f"PP:{horse['prime_power']:.1f}" if horse['prime_power'] > 0 else "FT"
                print(f"✓ Saved #{horse['horse_num']} {horse['horse_name']:20s} - Finished {horse['finish_position']} ({pp_str})")
            else:
                pp_str = f"PP:{horse['prime_power']:.1f}" if horse['prime_power'] > 0 else "FT"
                print(f"  Saved #{horse['horse_num']} {horse['horse_name']:20s} - DNF top 5 ({pp_str})")
        
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
