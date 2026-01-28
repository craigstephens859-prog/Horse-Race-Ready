"""
Bulk Race Data Entry Tool
Quick way to build historical database from race cards + results

Usage:
1. python bulk_race_entry.py
2. Enter race info + horse names
3. Enter finishing order (e.g., "6,3,8,2,1")
4. Repeat for multiple races
5. Check accuracy stats
"""

import sqlite3
from datetime import datetime
from ml_engine import RaceDatabase

def quick_entry():
    """Interactive prompt for entering race data."""
    db = RaceDatabase()
    
    print("\n🏇 BULK RACE DATA ENTRY TOOL 🏇\n")
    print("=" * 60)
    
    while True:
        print("\n" + "=" * 60)
        print("RACE INFORMATION")
        print("=" * 60)
        
        # Race details
        track = input("Track name (or 'done' to finish): ").strip()
        if track.lower() == 'done':
            break
        
        date = input("Date (YYYY-MM-DD, or press Enter for today): ").strip()
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        race_num = int(input("Race number: ").strip())
        distance = input("Distance (e.g., '6 Furlongs', '1 Mile'): ").strip()
        surface = input("Surface (Dirt/Turf/Synthetic): ").strip()
        
        # Create race
        race_data = {
            'track': track,
            'date': date,
            'race_number': race_num,
            'distance': distance,
            'surface': surface,
            'condition': 'fast',
            'race_type': 'unknown',
            'purse': 0,
            'field_size': 0
        }
        
        print("\n" + "-" * 60)
        print("HORSES (Enter horse names and odds)")
        print("-" * 60)
        
        horses = []
        post = 1
        
        while True:
            horse_name = input(f"Horse #{post} name (or press Enter if done): ").strip()
            if not horse_name:
                break
            
            odds_input = input(f"  Final odds for {horse_name} (decimal, e.g., 5.2): ").strip()
            try:
                odds = float(odds_input)
            except:
                odds = 10.0
            
            horses.append({
                'horse_name': horse_name,
                'post_position': post,
                'final_odds': odds,
                'predicted_win_prob': 1.0 / odds,
                'rating_total': 0.0
            })
            
            post += 1
        
        if not horses:
            print("⚠️ No horses entered, skipping race")
            continue
        
        race_data['field_size'] = len(horses)
        
        # Get finishing order
        print("\n" + "-" * 60)
        print("RESULTS (Enter finishing order)")
        print("-" * 60)
        print("Format: Enter horse names in order (1st, 2nd, 3rd, etc.)")
        print("Example: If #6 won, #3 second, #8 third, enter: 6,3,8,2,1")
        print("\nOr enter horse names: Horse A, Horse B, Horse C")
        
        finish_input = input("\nFinishing order: ").strip()
        
        # Parse finishing order
        if ',' in finish_input:
            finish_order = [x.strip() for x in finish_input.split(',')]
        else:
            finish_order = finish_input.split()
        
        # Try to match by post position or name
        results = []
        for position, entry in enumerate(finish_order, 1):
            # Try as post position first
            if entry.isdigit():
                post_num = int(entry)
                for h in horses:
                    if h['post_position'] == post_num:
                        results.append((h['horse_name'], position))
                        break
            else:
                # Try as horse name
                for h in horses:
                    if h['horse_name'].lower() == entry.lower():
                        results.append((h['horse_name'], position))
                        break
        
        if not results:
            print("⚠️ Could not parse results, skipping race")
            continue
        
        # Save to database
        try:
            race_id = db.save_race(race_data)
            db.save_horse_predictions(race_id, horses)
            db.update_race_results(race_id, results)
            
            print(f"\n✅ Race saved! Race ID: {race_id}")
            print(f"   Winner: {results[0][0]}")
            
            # Show quick stats
            summary = db.get_race_summary()
            print(f"\n📊 Total races in database: {summary['total_races']}")
            
        except Exception as e:
            print(f"\n❌ Error saving race: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    summary = db.get_race_summary()
    print(f"Total races entered: {summary['total_races']}")
    print(f"Total horses: {summary['total_horses']}")
    print(f"\nDatabase ready for ML training!")
    print("Run 'Train Model' in Section E when you have 10+ races.")
    print("=" * 60)

if __name__ == "__main__":
    quick_entry()
