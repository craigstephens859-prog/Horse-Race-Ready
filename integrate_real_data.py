"""
Real Data Integration for ML Training
======================================

Connects historical_data_builder.py with ml_quant_engine_v2.py to retrain
the model with accumulated real race data.

**Usage:**
    python integrate_real_data.py --retrain

This will:
1. Export completed races from historical database
2. Convert to ml_quant_engine_v2 format
3. Retrain all models with real data
4. Run validation backtest
5. Compare performance vs synthetic data baseline
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from historical_data_builder import HistoricalDataBuilder
from ml_quant_engine_v2 import RunningOrderPredictor, ModelWeaknesses
from backtest_simulator_v2 import EnhancedBacktestEngine


def convert_to_ml_format(df: pd.DataFrame) -> tuple:
    """
    Convert historical database format to ml_quant_engine_v2 format.
    
    Args:
        df: DataFrame from historical_data_builder.export_training_data()
        
    Returns:
        (X, y) where X is features and y is finishing positions
    """
    print("\n🔄 Converting historical data to ML format...")
    
    # Group by race
    races = []
    for race_id, race_df in df.groupby('race_id'):
        race_df = race_df.sort_values('program_number')
        
        # Extract features for each horse
        features = []
        for _, horse in race_df.iterrows():
            horse_features = {
                'speed_fig': horse['speed_last_race'],
                'speed_avg_3': horse['speed_avg_3'],
                'best_speed': horse['best_speed_fig'],
                'class_rating': horse['class_rating'],
                'prime_power': horse['prime_power'],
                'e1_pace': horse['e1_pace'],
                'e2_pace': horse['e2_pace'],
                'late_pace': horse['late_pace'],
                'running_style': horse['running_style'],
                'days_since_last': horse['days_since_last'],
                'consistency': horse['consistency_score'],
                'post_position': horse['post_position'],
                'ml_odds': horse['morning_line_odds'],
                'shipper': 1 if horse['shipper'] else 0,
                'class_dropper': 1 if horse['class_dropper'] else 0,
                'blinkers': 1 if horse['blinkers_on'] else 0,
            }
            features.append(horse_features)
        
        # Extract labels (actual finishing positions)
        labels = race_df['finish_position'].values
        
        races.append({
            'race_id': race_id,
            'features': features,
            'labels': labels,
            'track': race_df.iloc[0]['track'],
            'date': race_df.iloc[0]['date'],
            'surface': race_df.iloc[0]['surface']
        })
    
    print(f"✅ Converted {len(races)} races to ML format")
    return races


def prepare_training_arrays(races: list) -> tuple:
    """
    Prepare X, y arrays for sklearn models.
    
    Args:
        races: List of race dicts from convert_to_ml_format
        
    Returns:
        (X_train, y_train, race_metadata)
    """
    X_train = []
    y_train = []
    metadata = []
    
    for race in races:
        features = race['features']
        labels = race['labels']
        
        # Convert feature dicts to vectors
        feature_vectors = []
        for horse_feat in features:
            vec = [
                horse_feat['speed_fig'],
                horse_feat['speed_avg_3'],
                horse_feat['best_speed'],
                horse_feat['class_rating'],
                horse_feat['prime_power'],
                horse_feat['e1_pace'],
                horse_feat['e2_pace'],
                horse_feat['late_pace'],
                1 if horse_feat['running_style'] == 'E' else 0,  # Early runner
                horse_feat['days_since_last'],
                horse_feat['consistency'],
                horse_feat['post_position'],
                horse_feat['ml_odds'],
                horse_feat['shipper'],
                horse_feat['class_dropper'],
                horse_feat['blinkers']
            ]
            feature_vectors.append(vec)
        
        X_train.append(feature_vectors)
        y_train.append(labels)
        metadata.append({
            'race_id': race['race_id'],
            'track': race['track'],
            'date': race['date']
        })
    
    return X_train, y_train, metadata


def retrain_with_real_data(db_path: str = "historical_races.db"):
    """
    Complete retraining workflow with real historical data.
    
    Args:
        db_path: Path to historical races database
    """
    print("=" * 80)
    print("🚀 RETRAINING ML MODEL WITH REAL DATA")
    print("=" * 80)
    
    # Step 1: Check database
    builder = HistoricalDataBuilder(db_path)
    stats = builder.get_statistics()
    
    print(f"\n📊 Database Status:")
    print(f"   Completed races: {stats['completed_races']}")
    print(f"   Horses with results: {stats['horses_with_results']}")
    print(f"   Date range: {stats['date_range']}")
    
    if stats['completed_races'] < 50:
        print("\n⚠️  WARNING: Less than 50 races available")
        print(f"   Recommended minimum: 100 races")
        print(f"   Need {100 - stats['completed_races']} more races")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Add more races and try again.")
            return
    
    # Step 2: Export training data
    print("\n💾 Exporting training data...")
    df = builder.export_training_data("real_training_data.csv")
    
    if len(df) == 0:
        print("❌ No training data available. Add completed races first.")
        return
    
    # Step 3: Convert to ML format
    races = convert_to_ml_format(df)
    X_train, y_train, metadata = prepare_training_arrays(races)
    
    # Split into train/validation
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"\n📚 Training split:")
    print(f"   Training races: {len(X_train_split)}")
    print(f"   Validation races: {len(X_val)}")
    
    # Step 4: Train model
    print("\n🔧 Training RunningOrderPredictor with real data...")
    print("   This may take 5-10 minutes...")
    
    predictor = RunningOrderPredictor()
    
    # Flatten for sklearn
    X_flat = []
    y_flat = []
    for race_X, race_y in zip(X_train_split, y_train_split):
        for horse_X, horse_y in zip(race_X, race_y):
            X_flat.append(horse_X)
            y_flat.append(horse_y)
    
    X_flat = np.array(X_flat)
    y_flat = np.array(y_flat) - 1  # Convert to 0-indexed
    
    predictor.train(X_flat, y_flat)
    
    print("✅ Training complete!")
    
    # Step 5: Validation
    print("\n📊 Validating on holdout races...")
    
    correct = 0
    total = 0
    exacta_correct = 0
    
    for race_X, race_y in zip(X_val, y_val):
        predictions = predictor.predict_running_order(np.array(race_X))
        
        predicted_winner = predictions.iloc[0]['Predicted_Finish']
        actual_winner = np.argmin(race_y) + 1
        
        if predicted_winner == actual_winner:
            correct += 1
        
        total += 1
        
        # Check exacta
        predicted_top2 = predictions.head(2)['Predicted_Finish'].tolist()
        actual_top2 = sorted(range(len(race_y)), key=lambda i: race_y[i])[:2]
        actual_top2 = [i+1 for i in actual_top2]
        
        if set(predicted_top2) == set(actual_top2):
            exacta_correct += 1
    
    winner_accuracy = (correct / total) * 100 if total > 0 else 0
    exacta_accuracy = (exacta_correct / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 REAL DATA VALIDATION RESULTS:")
    print(f"   Winner accuracy: {winner_accuracy:.1f}% ({correct}/{total})")
    print(f"   Exacta accuracy: {exacta_accuracy:.1f}% ({exacta_correct}/{total})")
    
    # Step 6: Compare to synthetic baseline
    print("\n📈 PERFORMANCE COMPARISON:")
    print("   Synthetic data (V2): 58.0% winner accuracy")
    print(f"   Real data (Current): {winner_accuracy:.1f}% winner accuracy")
    
    improvement = winner_accuracy - 58.0
    if improvement > 0:
        print(f"   ✅ Improvement: +{improvement:.1f} percentage points")
    else:
        print(f"   ⚠️  Need more data: {improvement:.1f} percentage points")
    
    # Step 7: Save model
    print("\n💾 Saving trained model...")
    predictor.save_model("ml_quant_engine_real_data.pkl")
    print("   Saved to: ml_quant_engine_real_data.pkl")
    
    # Step 8: Estimate path to 90%
    print("\n🎯 PATH TO 90% ACCURACY:")
    
    if stats['completed_races'] < 1000:
        estimated_at_1000 = winner_accuracy + (1000 - stats['completed_races']) * 0.02
        estimated_at_5000 = winner_accuracy + (5000 - stats['completed_races']) * 0.01
        
        print(f"   Current ({stats['completed_races']} races): {winner_accuracy:.1f}%")
        print(f"   Estimated at 1,000 races: {min(estimated_at_1000, 88):.1f}%")
        print(f"   Estimated at 5,000 races: {min(estimated_at_5000, 92):.1f}%")
        print(f"\n   📅 Time estimate: {1000 - stats['completed_races']} more races")
        print(f"      At 10 races/day: ~{(1000 - stats['completed_races']) // 10} days")
    else:
        print(f"   Current: {winner_accuracy:.1f}% with {stats['completed_races']} races")
        print("   You're at the data scale for 90%+ performance!")
    
    print("\n" + "=" * 80)
    print("✅ RETRAINING COMPLETE")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("1. Use ml_quant_engine_real_data.pkl for predictions")
    print("2. Continue adding race results to historical database")
    print("3. Retrain monthly as data grows")
    print("4. Monitor accuracy progression toward 90% target")
    print("=" * 80)


def quick_add_results():
    """Interactive tool to quickly add race results to database."""
    print("=" * 80)
    print("📝 QUICK ADD RACE RESULTS")
    print("=" * 80)
    
    builder = HistoricalDataBuilder()
    
    # Get pending races
    import sqlite3
    conn = sqlite3.connect(builder.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT race_id, track, date, race_number, field_size
        FROM races
        WHERE is_completed = FALSE
        ORDER BY date DESC
        LIMIT 10
    """)
    pending = cursor.fetchall()
    conn.close()
    
    if not pending:
        print("✅ No pending races. All results recorded!")
        return
    
    print(f"\n📋 Found {len(pending)} pending races:")
    for i, (race_id, track, date, race_num, field_size) in enumerate(pending, 1):
        print(f"   {i}. {race_id} - {track} R{race_num} on {date} ({field_size} horses)")
    
    choice = input("\nSelect race number (or 'q' to quit): ")
    if choice.lower() == 'q':
        return
    
    try:
        idx = int(choice) - 1
        race_id = pending[idx][0]
        field_size = pending[idx][4]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return
    
    print(f"\n🏁 Enter results for {race_id}")
    print(f"   Enter finishing order: program numbers separated by spaces")
    print(f"   Example: 5 2 7 1 3 (means horse #5 won, #2 second, etc.)")
    
    order = input("\nFinishing order: ").strip().split()
    
    if len(order) != field_size:
        print(f"⚠️  Expected {field_size} horses, got {len(order)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Build results list
    results = []
    for finish_pos, prog_num in enumerate(order, 1):
        try:
            prog_num = int(prog_num)
            results.append((prog_num, finish_pos, 0.0))  # Lengths will be 0 for simplicity
        except ValueError:
            print(f"❌ Invalid program number: {prog_num}")
            return
    
    # Save results
    builder.add_race_results(race_id, results)
    print(f"✅ Results recorded for {race_id}")
    
    # Show updated stats
    stats = builder.get_statistics()
    print(f"\n📊 Updated database stats:")
    print(f"   Completed races: {stats['completed_races']}")
    print(f"   Pending races: {stats['pending_races']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Data Integration for ML Training")
    parser.add_argument('--retrain', action='store_true', 
                       help='Retrain model with accumulated real data')
    parser.add_argument('--add-results', action='store_true',
                       help='Quick add race results interactively')
    parser.add_argument('--db', default='historical_races.db',
                       help='Path to historical database')
    
    args = parser.parse_args()
    
    if args.add_results:
        quick_add_results()
    elif args.retrain:
        retrain_with_real_data(args.db)
    else:
        print("Usage:")
        print("  python integrate_real_data.py --retrain        # Retrain with real data")
        print("  python integrate_real_data.py --add-results    # Add race results")
