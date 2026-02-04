"""
VALIDATE PEGASUS WORLD CUP G1 PREDICTIONS
==========================================
Test unified rating engine with bug fix to verify correct predictions

Expected Results:
- Skippylongstocking (actual winner): 15-25%
- White Abarrio (2nd place): 18-25%
- Banishing: 3-7%

Actual Results: 5, 11, 3, 9, 2
"""

from unified_rating_engine import UnifiedRatingEngine
import pandas as pd

# Read full Pegasus PP text
with open('pegasus_wc_g1_pp.txt', 'r', encoding='utf-8') as f:
    pegasus_pp = f.read()

print("=" * 80)
print("PEGASUS WORLD CUP G1 VALIDATION TEST")
print("=" * 80)
print("\n🏇 Testing unified rating engine with all enhancements:")
print("   ✅ Bayesian uncertainty quantification")
print("   ✅ Multinomial logit finish probabilities")
print("   ✅ Bug fix: days_since_last attribute")
print()

# Initialize engine
engine = UnifiedRatingEngine(softmax_tau=3.0)

# Run prediction
try:
    results = engine.predict_race(
        pp_text=pegasus_pp,
        today_purse=3000000,
        today_race_type="Grade 1 Stakes",
        track_name="Gulfstream Park",
        surface_type="Dirt",
        distance_txt="1 1/8 Miles",
        condition_txt="fast"
    )
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    # Display main predictions
    display_cols = ['Horse', 'Post', 'Rating', 'Probability', 'Fair_Odds']
    
    # Add logit columns if available
    if 'P_Win_Logit' in results.columns:
        display_cols.extend(['P_Win_Logit', 'P_Place_Logit', 'P_Show_Logit', 'Expected_Finish_Logit'])
    
    print("\n" + results[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST ACTUAL RESULTS")
    print("=" * 80)
    
    actual_finish = {
        5: "Skippylongstocking (WINNER)",
        11: "White Abarrio (2nd)",
        3: "Full Serrano (3rd)",
        9: "Captain Cook (4th)",
        2: "British Isles (5th)"
    }
    
    print("\n📊 ACTUAL FINISH ORDER:")
    for post, info in actual_finish.items():
        print(f"   Post {post}: {info}")
    
    print("\n🎯 PREDICTIONS vs ACTUAL:")
    for post, info in actual_finish.items():
        horse_row = results[results['Post'] == post]
        if not horse_row.empty:
            horse_name = horse_row['Horse'].values[0]
            prob = horse_row['Probability'].values[0]
            predicted_pos = horse_row['Predicted_Finish'].values[0]
            
            if 'P_Win_Logit' in results.columns:
                p_win_logit = horse_row['P_Win_Logit'].values[0]
                exp_finish = horse_row['Expected_Finish_Logit'].values[0]
                print(f"\n   {horse_name} (Post {post}):")
                print(f"      Softmax Win Prob: {prob:.1%}")
                print(f"      Logit P(Win): {p_win_logit:.1%}")
                print(f"      Expected Finish: {exp_finish:.1f}")
                print(f"      Actual Finish: {list(actual_finish.keys()).index(post) + 1}")
            else:
                print(f"\n   {horse_name} (Post {post}):")
                print(f"      Win Probability: {prob:.1%}")
                print(f"      Predicted Finish: {predicted_pos}")
                print(f"      Actual Finish: {list(actual_finish.keys()).index(post) + 1}")
    
    # Key horses to check
    print("\n" + "=" * 80)
    print("KEY VALIDATION CHECKS")
    print("=" * 80)
    
    skippy = results[results['Post'] == 5]
    white_abarrio = results[results['Post'] == 11]
    banishing = results[results['Post'] == 4]
    
    print("\n✅ TARGET PROBABILITIES:")
    if not skippy.empty:
        skippy_prob = skippy['Probability'].values[0]
        target_met = "✅" if 0.15 <= skippy_prob <= 0.25 else "❌"
        print(f"   {target_met} Skippylongstocking: {skippy_prob:.1%} (target: 15-25%)")
    
    if not white_abarrio.empty:
        wa_prob = white_abarrio['Probability'].values[0]
        target_met = "✅" if 0.18 <= wa_prob <= 0.25 else "❌"
        print(f"   {target_met} White Abarrio: {wa_prob:.1%} (target: 18-25%)")
    
    if not banishing.empty:
        ban_prob = banishing['Probability'].values[0]
        target_met = "✅" if 0.03 <= ban_prob <= 0.07 else "❌"
        print(f"   {target_met} Banishing: {ban_prob:.1%} (target: 3-7%)")
    
    # Check if exotic probabilities available
    if hasattr(results, 'attrs') and 'exotic_probabilities' in results.attrs:
        exotics = results.attrs['exotic_probabilities']
        
        print("\n" + "=" * 80)
        print("EXOTIC BET PROBABILITIES")
        print("=" * 80)
        
        print("\n🎰 TOP 5 EXACTAS:")
        for i, (h1, h2, prob) in enumerate(exotics['exacta'][:5], 1):
            actual = "✅ ACTUAL RESULT!" if (h1 == "Skippylongstocking" and h2 == "White Abarrio") else ""
            print(f"   {i}. {h1}-{h2}: {prob:.2%} {actual}")
        
        print("\n🎰 TOP 5 TRIFECTAS:")
        for i, (h1, h2, h3, prob) in enumerate(exotics['trifecta'][:5], 1):
            actual = "✅ ACTUAL RESULT!" if (h1 == "Skippylongstocking" and h2 == "White Abarrio" and h3 == "Full Serrano") else ""
            print(f"   {i}. {h1}-{h2}-{h3}: {prob:.2%} {actual}")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    
    # Save results
    results.to_csv('pegasus_validation_results.csv', index=False)
    print("\n💾 Results saved to: pegasus_validation_results.csv")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
