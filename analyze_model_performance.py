"""
COMPREHENSIVE MODEL PERFORMANCE ANALYSIS
Analyzes predictions vs actual results from gold_high_iq.db
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

DB_PATH = "gold_high_iq.db"

def get_database_summary():
    """Get overall database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total races with results
    cursor.execute("SELECT COUNT(DISTINCT race_id) FROM gold_high_iq")
    total_races = cursor.fetchone()[0]
    
    # Total horses
    cursor.execute("SELECT COUNT(*) FROM gold_high_iq")
    total_horses = cursor.fetchone()[0]
    
    # Get all race IDs with results
    cursor.execute("""
        SELECT DISTINCT race_id 
        FROM gold_high_iq 
        ORDER BY race_id
    """)
    race_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        'total_races': total_races,
        'total_horses': total_horses,
        'race_ids': race_ids
    }

def calculate_accuracy_metrics():
    """Calculate winner, top-3, and top-5 accuracy"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get all races with complete data
    query = """
        SELECT 
            race_id,
            horse_name,
            program_number,
            actual_finish_position,
            predicted_finish_position as predicted_rank,
            prediction_error,
            timestamp as result_entered_timestamp
        FROM gold_high_iq
        ORDER BY race_id, actual_finish_position
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # Group by race
    races = df.groupby('race_id')
    
    winner_correct = 0
    top3_total_correct = 0
    top5_total_correct = 0
    total_races = len(races)
    
    race_details = []
    
    for race_id, race_df in races:
        # Sort by actual finish
        race_df = race_df.sort_values('actual_finish_position')
        
        # Get actual top 5
        actual_top5 = race_df.head(5)
        actual_winner = actual_top5.iloc[0]
        actual_top3_names = set(actual_top5.head(3)['horse_name'])
        actual_top5_names = set(actual_top5['horse_name'])
        
        # Get predicted top 5
        predicted_top5 = race_df.nsmallest(5, 'predicted_rank')
        predicted_winner = predicted_top5.iloc[0]
        predicted_top3_names = set(predicted_top5.head(3)['horse_name'])
        predicted_top5_names = set(predicted_top5['horse_name'])
        
        # Winner accuracy
        winner_hit = (predicted_winner['horse_name'] == actual_winner['horse_name'])
        if winner_hit:
            winner_correct += 1
        
        # Top-3 accuracy (how many of actual top 3 did we predict in top 3?)
        top3_hits = len(actual_top3_names & predicted_top3_names)
        top3_total_correct += top3_hits
        
        # Top-5 accuracy (how many of actual top 5 did we predict in top 5?)
        top5_hits = len(actual_top5_names & predicted_top5_names)
        top5_total_correct += top5_hits
        
        # Store details
        race_details.append({
            'race_id': race_id,
            'predicted_winner': predicted_winner['horse_name'],
            'predicted_winner_rank': predicted_winner['predicted_rank'],
            'actual_winner': actual_winner['horse_name'],
            'actual_winner_finish': actual_winner['actual_finish_position'],
            'winner_hit': winner_hit,
            'top3_hits': top3_hits,
            'top5_hits': top5_hits,
            'predicted_winner_actual_finish': race_df[race_df['horse_name'] == predicted_winner['horse_name']]['actual_finish_position'].values[0]
        })
    
    # Calculate percentages
    winner_accuracy = (winner_correct / total_races) * 100 if total_races > 0 else 0
    top3_accuracy = (top3_total_correct / (total_races * 3)) * 100 if total_races > 0 else 0
    top5_accuracy = (top5_total_correct / (total_races * 5)) * 100 if total_races > 0 else 0
    
    return {
        'total_races': total_races,
        'winner_correct': winner_correct,
        'winner_accuracy': winner_accuracy,
        'top3_correct': top3_total_correct,
        'top3_accuracy': top3_accuracy,
        'top5_correct': top5_total_correct,
        'top5_accuracy': top5_accuracy,
        'race_details': race_details
    }

def analyze_prediction_errors():
    """Analyze average prediction errors"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            race_id,
            horse_name,
            actual_finish_position,
            predicted_finish_position as predicted_rank,
            prediction_error
        FROM gold_high_iq
        ORDER BY race_id, actual_finish_position
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # Overall stats
    avg_error = df['prediction_error'].mean()
    median_error = df['prediction_error'].median()
    max_error = df['prediction_error'].max()
    min_error = df['prediction_error'].min()
    
    # Errors by finish position
    errors_by_position = df.groupby('actual_finish_position')['prediction_error'].agg(['mean', 'count']).reset_index()
    errors_by_position.columns = ['Position', 'Avg_Error', 'Count']
    
    return {
        'avg_error': avg_error,
        'median_error': median_error,
        'max_error': max_error,
        'min_error': min_error,
        'errors_by_position': errors_by_position
    }

def analyze_by_race_quality():
    """Analyze performance by race quality (claiming vs stakes)"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get race types from save scripts or infer from race_id
    query = """
        SELECT 
            race_id,
            horse_name,
            actual_finish_position,
            predicted_finish_position as predicted_rank,
            prediction_error
        FROM gold_high_iq
        ORDER BY race_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # Classify races (based on your save scripts)
    claiming_races = ['TUP_R4_20260202', 'TUP_R5_20260202']  # Both claiming
    stakes_races = ['SA_R4_20260201', 'SA_R6_20260201', 'SA_R8_20260201']  # SA R4 maiden, R6 allowance, R8 stakes
    turf_race = ['SA_R5_20260201']  # Turf maiden
    
    claiming_df = df[df['race_id'].isin(claiming_races)]
    stakes_df = df[df['race_id'].isin(stakes_races)]
    
    results = {}
    
    if not claiming_df.empty:
        results['claiming'] = {
            'count': len(claiming_df['race_id'].unique()),
            'avg_error': claiming_df['prediction_error'].mean(),
            'races': claiming_races
        }
    
    if not stakes_df.empty:
        results['stakes'] = {
            'count': len(stakes_df['race_id'].unique()),
            'avg_error': stakes_df['prediction_error'].mean(),
            'races': stakes_races
        }
    
    return results

def print_report():
    """Generate and print comprehensive report"""
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS - GOLD HIGH-IQ SYSTEM")
    print("=" * 80)
    print()
    
    # Database summary
    print("📊 DATABASE SUMMARY")
    print("-" * 80)
    summary = get_database_summary()
    print(f"Total Races: {summary['total_races']}")
    print(f"Total Horses: {summary['total_horses']}")
    print(f"Training Examples per Race: 5 (top 5 finishers)")
    print()
    print("Races with Results:")
    for race_id in summary['race_ids']:
        print(f"  • {race_id}")
    print()
    
    # Accuracy metrics
    print("🎯 ACCURACY METRICS")
    print("-" * 80)
    accuracy = calculate_accuracy_metrics()
    
    if accuracy:
        print(f"Winner Accuracy:  {accuracy['winner_correct']}/{accuracy['total_races']} = {accuracy['winner_accuracy']:.1f}%")
        print(f"Top-3 Accuracy:   {accuracy['top3_correct']}/{accuracy['total_races'] * 3} = {accuracy['top3_accuracy']:.1f}%")
        print(f"Top-5 Accuracy:   {accuracy['top5_correct']}/{accuracy['total_races'] * 5} = {accuracy['top5_accuracy']:.1f}%")
        print()
        
        # Race-by-race breakdown
        print("📋 RACE-BY-RACE RESULTS")
        print("-" * 80)
        for detail in accuracy['race_details']:
            status = "✅ HIT" if detail['winner_hit'] else f"❌ MISS (finished {detail['predicted_winner_actual_finish']})"
            print(f"\n{detail['race_id']}:")
            print(f"  Predicted: #{detail['predicted_winner_rank']} {detail['predicted_winner']}")
            print(f"  Actual:    WON - {detail['actual_winner']}")
            print(f"  Result:    {status}")
            print(f"  Top-3:     {detail['top3_hits']}/3 correct")
            print(f"  Top-5:     {detail['top5_hits']}/5 correct")
    else:
        print("No data available for accuracy analysis")
    print()
    
    # Prediction errors
    print("📉 PREDICTION ERROR ANALYSIS")
    print("-" * 80)
    errors = analyze_prediction_errors()
    
    if errors:
        print(f"Average Error:  {errors['avg_error']:.2f} positions")
        print(f"Median Error:   {errors['median_error']:.2f} positions")
        print(f"Min Error:      {errors['min_error']:.2f} positions")
        print(f"Max Error:      {errors['max_error']:.2f} positions")
        print()
        print("Average Error by Finish Position:")
        print(errors['errors_by_position'].to_string(index=False))
    else:
        print("No data available for error analysis")
    print()
    
    # Race quality analysis
    print("🏆 PERFORMANCE BY RACE QUALITY")
    print("-" * 80)
    quality = analyze_by_race_quality()
    
    if quality:
        if 'claiming' in quality:
            print(f"\nClaiming Races ({quality['claiming']['count']} races):")
            print(f"  Average Error: {quality['claiming']['avg_error']:.2f} positions")
            for race in quality['claiming']['races']:
                print(f"    • {race}")
        
        if 'stakes' in quality:
            print(f"\nStakes/Allowance Races ({quality['stakes']['count']} races):")
            print(f"  Average Error: {quality['stakes']['avg_error']:.2f} positions")
            for race in quality['stakes']['races']:
                print(f"    • {race}")
        
        if 'claiming' in quality and 'stakes' in quality:
            claiming_error = quality['claiming']['avg_error']
            stakes_error = quality['stakes']['avg_error']
            diff = claiming_error - stakes_error
            print(f"\n📊 Error Difference: {abs(diff):.2f} positions")
            if claiming_error > stakes_error:
                print(f"   Claiming races have {diff:.2f} more error (more chaotic)")
            else:
                print(f"   Stakes races have {abs(diff):.2f} more error")
    else:
        print("No data available for race quality analysis")
    print()
    
    # Summary insights
    print("=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)
    
    if accuracy and errors:
        print(f"• Model is picking winners at {accuracy['winner_accuracy']:.1f}% accuracy")
        print(f"• Top-3 predictions capture {accuracy['top3_accuracy']:.1f}% of trifecta horses")
        print(f"• Top-5 predictions capture {accuracy['top5_accuracy']:.1f}% of superfecta horses")
        print(f"• Average prediction error is {errors['avg_error']:.2f} positions")
        
        if quality and 'claiming' in quality and 'stakes' in quality:
            if quality['claiming']['avg_error'] > quality['stakes']['avg_error']:
                print(f"• Claiming races are more unpredictable (+{quality['claiming']['avg_error'] - quality['stakes']['avg_error']:.2f} error)")
                print("  → Recent fixes (62%/55% PP, 2.5x Speed boost) should improve this")
        
        print(f"\n🚀 NEXT MILESTONE: Need {50 - accuracy['total_races']} more races to reach 50 for model retraining")
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    try:
        print_report()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
