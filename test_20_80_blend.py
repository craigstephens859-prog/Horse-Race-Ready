"""Quick test of 20/80 blend"""
import pandas as pd

def calc_rating(row):
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    comp_score = class_norm * 3.0 + form_norm * 1.8 + speed_norm * 1.8 + pace_norm * 1.5
    pp_norm = (row['prime_power'] - 110) / 20
    pp_score = pp_norm * 10
    return 0.20 * comp_score + 0.80 * pp_score

horses_r8 = {
    'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'speed_last': [91, 78, 79, 77, 77, 81, 80, 73, 65, 76, 72, 82, 77],
    'speed_best_dist': [91, 89, 82, 84, 88, 81, 88, 93, 83, 82, 87, 89, 83],
    'prime_power': [117.7, 124.4, 123.5, 120.4, 122.3, 121.0, 118.1, 125.4, 114.6, 116.6, 119.0, 127.5, 125.3],
    'class_rating': [113.3, 114.4, 114.6, 112.5, 113.6, 112.5, 112.3, 113.9, 111.3, 112.5, 112.8, 114.8, 113.5],
    'e1_pace': [99, 95, 78, 93, 88, 94, 88, 87, 87, 88, 84, 85, 92],
    'actual_finish': [99, 99, 99, 99, 4, 99, 99, 3, 5, 99, 99, 2, 1]
}
df = pd.DataFrame(horses_r8)
df['rating'] = df.apply(calc_rating, axis=1)
df_sorted = df.sort_values('rating', ascending=False)

print('SA R8 with 20/80 blend (20% Components / 80% Prime Power):')
print('-' * 60)
for i in range(5):
    h = df_sorted.iloc[i]
    act = int(h['actual_finish']) if h['actual_finish'] < 99 else '--'
    print(f'Pred:{i+1} Act:{act:>2} | #{int(h["num"]):2d} PP:{h["prime_power"]:.1f}')

winner_in_top3 = 13 in df_sorted.head(3)['num'].values
print(f'\nWinner #13 in top 3: {winner_in_top3}')
