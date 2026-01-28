"""
Quick Demo: Kaggle Triple Crown Parser
======================================
Demonstrates parsing without actual Kaggle file.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def create_demo_kaggle_data():
    """Create synthetic Kaggle-like data for demo."""
    
    # Simulate 3 Triple Crown races
    demo_data = {
        'race_date': [
            '2023-05-06', '2023-05-06', '2023-05-06', '2023-05-06', '2023-05-06',
            '2023-05-20', '2023-05-20', '2023-05-20', '2023-05-20', '2023-05-20',
            '2023-06-10', '2023-06-10', '2023-06-10', '2023-06-10', '2023-06-10',
        ],
        'track_name': [
            'Churchill Downs', 'Churchill Downs', 'Churchill Downs', 'Churchill Downs', 'Churchill Downs',
            'Pimlico', 'Pimlico', 'Pimlico', 'Pimlico', 'Pimlico',
            'Belmont Park', 'Belmont Park', 'Belmont Park', 'Belmont Park', 'Belmont Park',
        ],
        'race_name': [
            'Kentucky Derby', 'Kentucky Derby', 'Kentucky Derby', 'Kentucky Derby', 'Kentucky Derby',
            'Preakness Stakes', 'Preakness Stakes', 'Preakness Stakes', 'Preakness Stakes', 'Preakness Stakes',
            'Belmont Stakes', 'Belmont Stakes', 'Belmont Stakes', 'Belmont Stakes', 'Belmont Stakes',
        ],
        'horse_name': [
            'Mage', 'Two Phil\'s', 'Angel of Empire', 'Disarm', 'Hit Show',
            'National Treasure', 'Blazing Sevens', 'Perform', 'Mage', 'Coffee',
            'Arcangelo', 'Forte', 'Tapit Trice', 'Kingsbarns', 'National Treasure',
        ],
        'post_position': [
            15, 2, 8, 9, 7,
            6, 4, 5, 3, 2,
            13, 1, 14, 11, 12,
        ],
        'odds': [
            15.0, 7.5, 5.5, 6.0, 12.0,
            2.5, 8.0, 15.0, 7.5, 20.0,
            17.0, 0.8, 5.5, 25.0, 6.0,
        ],
        'finish_position': [
            1, 2, 3, 4, 5,
            1, 2, 3, 4, 5,
            1, 2, 3, 4, 5,
        ]
    }
    
    return pd.DataFrame(demo_data)


def demo_parse():
    """Demo the parsing workflow."""
    
    print("="*70)
    print("KAGGLE TRIPLE CROWN PARSER - DEMO MODE")
    print("="*70)
    print()
    print("📝 NOTE: This demo uses synthetic data to show the parsing workflow.")
    print("   To use real Kaggle data:")
    print("   1. Download from: https://www.kaggle.com/datasets/jtsw/triple-crown-horse-racing")
    print("   2. Run: python kaggle_triple_crown_parser.py --file <csv_file>")
    print()
    print("="*70)
    print()
    
    # Create demo data
    print("📊 Creating demo dataset (3 races, 15 horses)...")
    df = create_demo_kaggle_data()
    print(f"✅ Loaded {len(df)} horses from {df['race_name'].nunique()} races")
    print()
    
    # Show sample
    print("📋 Sample Input Data:")
    print("-"*70)
    print(df[['race_date', 'track_name', 'horse_name', 'post_position', 'odds', 'finish_position']].head(10))
    print()
    
    # Convert to ML format
    print("🔄 Converting to ML training format...")
    print()
    
    sample_rows = []
    
    for idx, row in df.iterrows():
        # Estimate features (same logic as main parser)
        speed_fig = 108.0 if row['finish_position'] == 1 else 104.0 if row['finish_position'] == 2 else 101.0
        class_rating = 110.0  # Grade 1
        prime_power = 185.0 if row['odds'] < 3 else 175.0 if row['odds'] < 5 else 165.0
        e1_pace = 95.0 if row['post_position'] <= 3 else 90.0
        running_style = 1 if row['post_position'] <= 3 else 2 if row['post_position'] <= 7 else 3
        
        sample_row = {
            'Race_Date': row['race_date'],
            'Track': 'CD' if 'Churchill' in row['track_name'] else 'PIM' if 'Pimlico' in row['track_name'] else 'BEL',
            'Race_Name': row['race_name'],
            'Horse_Name': row['horse_name'],
            'Post_Position': row['post_position'],
            'ML_Odds': row['odds'],
            'Finish_Position': row['finish_position'],
            'Win_Label': 1 if row['finish_position'] == 1 else 0,
            'Speed_Fig': speed_fig,
            'Class_Rating': class_rating,
            'Prime_Power': prime_power,
            'E1_Pace': e1_pace,
            'Running_Style': running_style,
        }
        sample_rows.append(sample_row)
    
    sample_df = pd.DataFrame(sample_rows)
    
    print("✅ Converted to ML format!")
    print()
    print("📊 Output CSV Structure (First 10 Rows):")
    print("="*70)
    print(sample_df.head(10).to_string(index=False))
    print()
    
    # Show statistics
    print("="*70)
    print("📈 Dataset Statistics:")
    print("="*70)
    print(f"Total Races:      {sample_df['Race_Name'].nunique()}")
    print(f"Total Horses:     {len(sample_df)}")
    print(f"Winners:          {sample_df['Win_Label'].sum()}")
    print(f"Average Field:    {len(sample_df) / sample_df['Race_Name'].nunique():.1f} horses/race")
    print(f"Date Range:       {sample_df['Race_Date'].min()} to {sample_df['Race_Date'].max()}")
    print(f"Tracks:           {', '.join(sample_df['Track'].unique())}")
    print()
    
    # Generate PyTorch arrays
    print("🔢 Generating PyTorch Arrays...")
    print()
    
    feature_cols = ['Speed_Fig', 'Class_Rating', 'Prime_Power', 'E1_Pace', 'Running_Style', 
                    'Post_Position', 'ML_Odds']
    
    # Note: Real parser has 16 features, demo has 7 for simplicity
    X = sample_df[feature_cols].values.astype(np.float32)
    y = sample_df['Win_Label'].values.astype(np.float32)
    
    print(f"✅ Generated arrays:")
    print(f"   X shape: {X.shape} (horses × features)")
    print(f"   y shape: {y.shape} (win labels)")
    print(f"   Winners: {y.sum():.0f} / {len(y)} ({y.mean()*100:.1f}%)")
    print()
    
    # Show limitations
    print("="*70)
    print("⚠️  KAGGLE DATASET LIMITATIONS:")
    print("="*70)
    print("❌ Only 45 races total (3/year × 15 years)")
    print("❌ Only Grade 1 stakes (not representative of daily racing)")
    print("❌ Missing pace figures (E1/E2/Late estimated)")
    print("❌ Missing trip notes (trouble, wide, etc.)")
    print("❌ Missing workout data")
    print("❌ Missing equipment changes")
    print("❌ Insufficient for 90% accuracy (need 1,000+ races)")
    print()
    print("✅ USE AS SUPPLEMENT ONLY")
    print("✅ Your daily BRISNET captures via Section F are 10x more valuable!")
    print("="*70)
    print()
    
    # Integration note
    print("🔗 Integration with Your System:")
    print("-"*70)
    print("After downloading real Kaggle data, you can:")
    print("1. Run: python kaggle_triple_crown_parser.py --file triple_crown.csv")
    print("2. Get: kaggle_triple_crown_X.npy and kaggle_triple_crown_y.npy")
    print("3. Load in ml_quant_engine_v2.py for supplemental training")
    print()
    print("Example code:")
    print("""
    import numpy as np
    
    # Load Kaggle data
    X_kaggle = np.load('kaggle_triple_crown_X.npy')
    y_kaggle = np.load('kaggle_triple_crown_y.npy')
    
    # Load your real data
    from integrate_real_data import prepare_training_arrays
    X_real, y_real = prepare_training_arrays('historical_races.db')
    
    # Combine (Kaggle as supplement)
    X_combined = np.vstack([X_real, X_kaggle])
    y_combined = np.hstack([y_real, y_kaggle])
    
    print(f"Real data: {len(X_real)} races")
    print(f"Kaggle supplement: {len(X_kaggle)} races")
    print(f"Total: {len(X_combined)} races")
    """)
    print()
    print("="*70)
    print("✅ Demo Complete!")
    print("="*70)


if __name__ == '__main__':
    demo_parse()
