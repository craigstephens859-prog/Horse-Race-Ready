import pandas as pd
import numpy as np

# Race data from the report
horses = {
    '4_Windribbon': {'class': -0.50, 'speed': 0.96, 'form': 1.75, 'pace': 0.49, 'style': 0.40, 'post': 0.00, 'track': 0.00, 'quirin': 5, 'actual': 1},
    '7_Big_Cheeseola': {'class': 1.00, 'speed': -0.59, 'form': 0.80, 'pace': 0.49, 'style': 0.10, 'post': 0.00, 'track': 0.00, 'quirin': 3, 'actual': 2},
    '3_Smarty_Nose': {'class': -0.30, 'speed': 0.96, 'form': 1.65, 'pace': 0.26, 'style': -0.30, 'post': 0.30, 'track': 0.00, 'quirin': 0, 'actual': 4},
    '8_Carols_Comic': {'class': -0.30, 'speed': -0.59, 'form': 0.80, 'pace': 0.26, 'style': -0.30, 'post': 0.00, 'track': 0.00, 'quirin': 0, 'actual': 3},
    '2_Elegant_Life': {'class': -0.30, 'speed': 0.18, 'form': 1.65, 'pace': 0.26, 'style': -0.30, 'post': 0.30, 'track': 0.00, 'quirin': 3, 'actual': 5}
}

# Weight schemes to test
schemes = {
    'Current': {'class': 2.5, 'speed': 2.0, 'form': 1.8, 'pace': 1.5, 'style': 1.2, 'post': 0.8},
    'Class_Up': {'class': 3.0, 'speed': 1.8, 'form': 1.8, 'pace': 1.5, 'style': 1.2, 'post': 0.8},
    'Speed_Down': {'class': 2.5, 'speed': 1.5, 'form': 1.8, 'pace': 1.5, 'style': 1.2, 'post': 0.8},
    'Class_Speed_Swap': {'class': 3.0, 'speed': 1.5, 'form': 1.8, 'pace': 1.5, 'style': 1.2, 'post': 0.8},
    'Balanced': {'class': 2.0, 'speed': 2.0, 'form': 2.0, 'pace': 1.5, 'style': 1.2, 'post': 0.8},
    'With_Quirin': {'class': 2.5, 'speed': 2.0, 'form': 1.8, 'pace': 1.5, 'style': 1.2, 'post': 0.8, 'quirin': 0.15}
}

print('=' * 80)
print('WEIGHT SCHEME ANALYSIS - Santa Anita R6 (Actual: 4,7,8,3,2)')
print('=' * 80)

for scheme_name, weights in schemes.items():
    print(f'\n{scheme_name} Weights: Class={weights["class"]:.1f}x Speed={weights["speed"]:.1f}x Form={weights["form"]:.1f}x', end='')
    if 'quirin' in weights:
        print(f' Quirin={weights["quirin"]:.2f}x/pt')
    else:
        print()
    print('-' * 80)
    
    results = []
    for horse_name, data in horses.items():
        rating = (
            data['class'] * weights['class'] +
            data['speed'] * weights['speed'] +
            data['form'] * weights['form'] +
            data['pace'] * weights['pace'] +
            data['style'] * weights['style'] +
            data['post'] * weights['post'] +
            data['track']
        )
        
        # Add Quirin if in scheme
        if 'quirin' in weights:
            rating += data['quirin'] * weights['quirin']
        
        results.append({
            'Horse': horse_name,
            'Rating': rating,
            'Actual': data['actual'],
            'Class_Contrib': data['class'] * weights['class'],
            'Speed_Contrib': data['speed'] * weights['speed'],
            'Form_Contrib': data['form'] * weights['form']
        })
    
    # Sort by rating
    results_df = pd.DataFrame(results).sort_values('Rating', ascending=False)
    
    for idx, row in results_df.iterrows():
        position = list(results_df.index).index(idx) + 1
        actual = int(row['Actual'])
        marker = '✓' if position == actual else '✗'
        print(f'{marker} Pred:{position} Act:{actual} | {row["Horse"]:20s} Rating:{row["Rating"]:6.2f} (Cls:{row["Class_Contrib"]:+5.2f} Spd:{row["Speed_Contrib"]:+5.2f} Frm:{row["Form_Contrib"]:+5.2f})')
    
    # Calculate accuracy score
    correct = sum(1 for idx, row in results_df.iterrows() if (list(results_df.index).index(idx) + 1) == int(row['Actual']))
    print(f'\n✓ Exact position matches: {correct}/5')

print('\n' + '=' * 80)
print('KEY INSIGHT:')
print('=' * 80)
print('Current system: Class 2.5x, Speed 2.0x')
print('#7 Class advantage: +1.00 * 2.5 = +2.50')
print('#3 Speed advantage: +0.96 * 2.0 = +1.92')
print('#7 was 0.58 points BEHIND #3 despite finishing 2nd in reality')
print('\nThe class gap: #7 (+1.00) vs #3 (-0.30) = +1.30 raw difference')
print('The speed gap: #3 (+0.96) vs #7 (-0.59) = +1.55 raw difference')
print('\nFor #7 to rank ahead of #3:')
print('  Class weight * 1.30 > Speed weight * 1.55')
print('  Class weight > 1.19 * Speed weight')
print('  If Speed = 2.0, then Class must be > 2.38')
print('  If Speed = 1.5, then Class can be 1.79+')
