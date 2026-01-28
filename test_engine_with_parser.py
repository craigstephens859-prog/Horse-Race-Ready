"""Comprehensive test of parser improvements with unified rating engine"""

from elite_parser import EliteBRISNETParser
from unified_rating_engine import UnifiedRatingEngine

# Sample BRISNET PP text for two horses
test_pp = """
1 Brown Sugar (E/P 1)
Own: Resolute Racing
ORTIZ JOSE L (0 0-0-0 0%)
B. f. 4
Sire : Twirling Candy (Candy Ride (ARG)) $60,000
Dam: Cashmere (Cowboy Cal)
Trnr: Joseph, Jr. Saffie A (103 19-18-18 18%)
Prime Power: 127.6 (9th) Life: 7 2 - 1 - 2 $74,450 86
Sire Stats: AWD 6.8 17%Mud 1420MudSts 1.69spi
Dam'sSire: AWD 6.8 11%Mud 596MudSts 0.86spi
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
13Dec25GP¯ à 1ˆ fm :23¨ :46ª1:10¨ 1:40 ¨¨¬ ™TrPOaksL 125k¨¨¬ 83 87/ 84 +11 +7 86 12 11ƒ 12®ƒ 11¬ 9¬ 6«ƒ HusbandsMJ¨¨¯ 16.30 Top3
19Aug25Prx¨¨ 1Ñ ft :23¨ :46ª1:10ª 1:40ª ¨¨¬ ™CathSphiaL 200k¨¨ª 59 51/ 99 -7 0 72 10 13° 13¨© 12©© 12¨ 12¨° OrtizJL¨©¨ 5.10 Top3
01Jly25Ind« 1m gd :23© :46ª1:12 1:37ª ¦ ¨¨¨ ™A34000n2x ¨¨¬ 64 74/ 90 -3 -6 80 7 5«ƒ 4® 1¨ 1¨ 1ƒ PedrozaM¨¨° L *1.10 Winner
29May25Ind¬ š 1ˆ ft :25 :50¨1:14« 1:46¨ ¦ ¨§° ™Mdn 32k ¨¨ª 65 68/ 97 -18 -14 83 5 2¨ 2 2 1¨ 1ª PedrozaM¨©§ L *0.05 Winner
12Apr25GP à 1ˆ fm :23 :46ª1:11© 1:40« ¦ ¨¨¨ ™Mdn 55k ¨¨© 75 69/ 99 +1 -10 84 3 7ªƒ 8¬ 8ª‚ 3¨ 3¨ BravoJ¨¨¯ L *1.70 Top3

2 Miss Mary Nell (E/P 5)
Own: Palm Beach Racing V Llc
BRAVO JOE (46 5-5-8 11%)
Dkbbr. f. 4
Sire : Girvin (Tale of Ekati) $25,000
Dam: Tempest's Flash (It's No Joke)
Trnr: David Carlos A (58 19-7-6 33%)
Prime Power: 127.6 (10th) Life: 16 5 - 3 - 2 $227,913 90
Sire Stats: AWD 6.5 22%Mud 221MudSts 1.32spi
Dam'sSire: AWD 7.2 18%Mud 383MudSts 0.61spi
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
10Jan26GP¨¨ à 1m fm :23© :46©1:09© 1:33¨ ¡ ¨¨« ™'SnshnFMTfB 75k¨¨ª 86 87/ 83 +11 +10 84 3 4ª 5 7 6« 7¬‚ RosarioJ¨©§ bf 17.70 Midpack
13Dec25GP¯ à 1ˆ fm :23¨ :46ª1:10¨ 1:40 ¨¨¬ ™TrPOaksL 125k¨¨« 99 98/ 69 +11 +7 83 11 1² 1 1 2 8® GonzalezE¨¨¯ bf 63.10 Top2
11Jly25GP¯ Ì 1Ñ ft :24¨ :48 1:12¨ 1:41ª ¨¨ª ™OC75k/n1x-v¨¨¬ 94 96/ 83 +7 +6 90 1 1¨ 1¨ 1¨ 1© 1¬‚ GonzalezE¨©§ Lb *0.90 Winner
07Jun25GP° à 1ˆ fm :22ª :46«1:09ª 1:40© ¨¨ª ™MrthWshtnB 75k¨¨« 88 101/ 71 +5 +12 86 6 4¨ 3© 2¨ 2ª 2« GonzalezE¨©§ b 14.60 Place
09May25GP¯ Ì 1ˆ ft :24ª :49©1:13 1:43 ¨¨ª Alw58000nc ¨¨© 73 78/ 87 -10 -5 83 1 3© 3© 4©ƒ 4« 2¬ ReyesL¨¨¯ Lb 2.20 Place
"""

print("=" * 80)
print("UNIFIED RATING ENGINE TEST WITH IMPROVED PARSER")
print("=" * 80)

# Initialize engine
engine = UnifiedRatingEngine()

# Parse horses
print("\n1. PARSING HORSES:")
print("-" * 60)
horses = engine.parser.parse_full_pp(test_pp)
for name, horse in horses.items():
    print(f"\n{name}:")
    print(f"  Post: {horse.post}")
    print(f"  Speed Figures: {horse.speed_figures[:5]}")
    print(f"  Avg Top 2: {horse.avg_top2}")
    print(f"  Recent Purses: {horse.recent_purses[:5]}")
    print(f"  Race Types: {horse.race_types[:5]}")
    print(f"  Recent Finishes: {horse.recent_finishes[:5]}")

# Predict race with $100k purse
print("\n2. COMPONENT RATINGS CALCULATION:")
print("-" * 60)
print("Race: $100,000 Handicap at 1m70yds")
print()

try:
    df = engine.predict_race(
        pp_text=test_pp,
        today_purse=100000,
        today_race_type="Hcp",
        track_name="GP",
        surface_type="D",
        distance_txt="1m70yds"
    )
    
    if not df.empty:
        print("Available columns:", df.columns.tolist())
        print("\nResults DataFrame (component ratings):")
        comp_cols = [c for c in ['Horse', 'Post', 'Cspeed', 'Cclass', 'Cform', 'Cpace', 'Cstyle', 'Cpost'] if c in df.columns]
        if comp_cols:
            print(df[comp_cols].to_string(index=False))
        
        print("\nResults DataFrame (predictions):")
        pred_cols = [c for c in ['Horse', 'Rating', 'WinProb', 'Confidence'] if c in df.columns]
        if pred_cols:
            print(df[pred_cols].to_string(index=False))
        
        print("\n3. COMPONENT RATING ANALYSIS:")
        print("-" * 60)
        for idx, row in df.iterrows():
            print(f"\n{row['Horse']}:")
            if 'Cspeed' in row: print(f"  C-Speed:  {row['Cspeed']:+.2f}  (Speed figure differential)")
            if 'Cclass' in row: print(f"  C-Class:  {row['Cclass']:+.2f}  (Purse/race type class)")
            if 'Cform' in row: print(f"  C-Form:   {row['Cform']:+.2f}  (Recent form cycle)")
            if 'Cpace' in row: print(f"  C-Pace:   {row['Cpace']:+.2f}  (Pace scenario)")
            if 'Cstyle' in row: print(f"  C-Style:  {row['Cstyle']:+.2f}  (Running style)")
            if 'Cpost' in row: print(f"  C-Post:   {row['Cpost']:+.2f}  (Post position)")
            print(f"  ----------------------------------------")
            if 'Rating' in row: print(f"  Rating:   {row['Rating']:.2f}")
            if 'WinProb' in row: print(f"  Win Prob: {row['WinProb']:.1f}%")
    else:
        print("ERROR: Empty DataFrame returned!")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE - Component ratings should now show differentiation!")
print("=" * 80)
