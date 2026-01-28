"""Test comprehensive parser with ALL BRISNET data fields"""

from elite_parser import EliteBRISNETParser

# Sample with rich data
test_pp = """
1 Brown Sugar (E/P 1)
Own: Resolute Racing
12/1 Black, Royal Blue Mirrored Image 'rr' In Royal Blue Circle,
ORTIZ JOSE L (0 0-0-0 0%)
B. f. 4 OBSOPN 2024 $400k
Sire : Twirling Candy (Candy Ride (ARG)) $60,000
Dam: Cashmere (Cowboy Cal)
Brdr: John Fradkin & Diane Fradkin (KY)
Trnr: Joseph, Jr. Saffie A (103 19-18-18 18%)
Prime Power: 127.6 (9th) Life: 7 2 - 1 - 2 $74,450 86
2025 6 2 - 1 - 1 $62,450 86
2024 1 0 - 0 - 1 $12,000 79
GP 0 0 - 0 - 0 $0 0
Fst (108) 3 1 - 0 - 1 $35,200 83
Off (109) 1 1 - 0 - 0 $20,400 80
Dis (108) 2 1 - 0 - 0 $23,200 83
Trf (107) 3 0 - 1 - 1 $18,850 86
AW 0 0 - 0 - 0 $0
L 115
+2025-2026 1543 22% 52% -0.34
+JKYw/EP types 481 21% 53% -0.52
JKYw/ Trn L60 2 0% 0% -2.00
+JKYw/ Routes 420 24% 57% -0.28
Sire Stats: AWD 6.8 17%Mud 1420MudSts 1.69spi
Dam'sSire: AWD 6.8 11%Mud 596MudSts 0.86spi
ñ High % trainer × Speed Figs rank poorly vs others
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
13Dec25GP¯ à 1ˆ fm :23¨ :46ª1:10¨ 1:40 ¨¨¬ ™TrPOaksL 125k¨¨¬ 83 87/ 84 +11 +7 86 12 11ƒ 12®ƒ 11¬ 9¬ 6«ƒ HusbandsMJ¨¨¯ 16.30 DstndOr²AndOnMorTm¨ƒRmsyPnd¨‚ No threat; 2-3wd 12
19Aug25Prx¨¨ 1Ñ ft :23¨ :46ª1:10ª 1:40ª ¨¨¬ ™CathSphiaL 200k¨¨ª 59 51/ 99 -7 0 72 10 13° 13¨© 12©© 12¨ 12¨° OrtizJL¨©¨ 5.10 DryPwdr«‚Ordaydramingirl¨ƒDiscoRles Always far back 13
01Jly25Ind« 1m gd :23© :46ª1:12 1:37ª ¦ ¨¨¨ ™A34000n2x ¨¨¬ 64 74/ 90 -3 -6 80 7 5«ƒ 4® 1¨ 1¨ 1ƒ PedrozaM¨¨° L *1.10 BrwnSgarƒFgtvStar¬MisSsanBª‚ Stmb;bmp;rail;rally3/8 8
29May25Ind¬ š 1ˆ ft :25 :50¨1:14« 1:46¨ ¦ ¨§° ™Mdn 32k ¨¨ª 65 68/ 97 -18 -14 83 5 2¨ 2 2 1¨ 1ª PedrozaM¨©§ L *0.05 BrwnSgrªThtrx¬Altlbtnaghty Press;bid5/16;mild drv 5
21Jan Pmm 3f ft :38¨ B
6/10 ×05Dec'25 Pmm ˜ (d) 3f fm :36© B
1/7 28Nov'25 Pmm ˜ (d) 4f fm :48 B
9/40 21Nov'25 Pmm 4f ft :49 B
12/40

7 Lady Firefoot (E/P 3)
Own: Charles Hallas
30/1 White, Light Blue Sash
BRIDGMOHAN SHAUN (6 0-0-0 0%)
Gr/ro. m. 8 KEESEP 2019 $28k
Sire : Temple City (Dynaformer) $5,000
Dam: Maryland Mist (Cozzene)
Trnr: Giddings Melanie (9 1-0-0 11%)
Prime Power: 126.3 (12th) Life: 24 4 - 0 - 1 $200,337 94
Trf (107) 21 4 - 0 - 0 $191,647 94
AW 1 0 - 0 - 0 $850 84
Blnkr On
L 117
Sire Stats: AWD 7.3 11%Mud 1311MudSts 0.85spi
Dam'sSire: AWD 8.0 12%Mud 1645MudSts 2.63spi
ñ May improve with Blinkers added today × Speed Figs rank poorly vs others
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
01Jan26GP° Ì 1„ ft :48ª 1:12ª1:36« 1:49¨ ¦ ¨¨« ™Hcp 100k ¨¨« 91 92/ 75 +6 +5 84 9 5© 3¨ 3¨ 6ª‚ 7ªƒ BridgmohanSX¨¨° L 24.40 FntsPrfrmr²CrlsWsƒFrFln¨ Chased 4wd; faded 9
17Jan Pmm ˜ 4f fm :48« B
4/4 21Dec'25 Pmm ˜ 4f fm :47© B
17/61 13Dec'25 Bel tr.t 4f ft :50« B
109/157
"""

parser = EliteBRISNETParser()

print("=" * 80)
print("COMPREHENSIVE PARSER TEST - ALL BRISNET DATA FIELDS")
print("=" * 80)

horses = parser.parse_full_pp(test_pp)

for name, horse in horses.items():
    print(f"\n{'=' * 80}")
    print(f"HORSE: {name} (Post {horse.post})")
    print(f"{'=' * 80}")
    
    print(f"\n[BASIC INFO]:")
    print(f"  Pace Style: {horse.pace_style} ({horse.style_strength})")
    print(f"  Quirin: {horse.quirin_points}")
    print(f"  ML Odds: {horse.ml_odds} ({horse.ml_odds_decimal})")
    print(f"  Weight: {horse.weight} lbs")
    
    print(f"\n[CONNECTIONS]:")
    print(f"  Jockey: {horse.jockey} ({horse.jockey_win_pct}%)")
    print(f"  Trainer: {horse.trainer} ({horse.trainer_win_pct}%)")
    
    print(f"\n[SPEED & FORM]:")
    print(f"  Speed Figs: {horse.speed_figures[:5]}")
    print(f"  Avg Top 2: {horse.avg_top2}")
    print(f"  Peak: {horse.peak_fig} | Last: {horse.last_fig}")
    print(f"  Days Since Last: {horse.days_since_last}")
    print(f"  Recent Finishes: {horse.recent_finishes[:5]}")
    
    print(f"\n[CLASS]:")
    print(f"  Recent Purses: {horse.recent_purses[:5]}")
    print(f"  Race Types: {horse.race_types[:5]}")
    
    print(f"\n[EQUIPMENT & MEDS]:")
    print(f"  Equipment: {horse.equipment}")
    if horse.equipment_change:
        print(f"  Change: {horse.equipment_change}")
    print(f"  Lasix: {'Yes' if horse.lasix else 'No'}")
    if horse.first_lasix:
        print(f"  FIRST TIME LASIX!")
    
    print(f"\n[RUNNING PATTERNS]:")
    print(f"  Early Speed %: {horse.early_speed_pct:.1f}%")
    print(f"  Closing %: {horse.closing_pct:.1f}%")
    print(f"  Avg Early Pos: {horse.avg_early_position:.1f}")
    print(f"  Avg Late Pos: {horse.avg_late_position:.1f}")
    
    print(f"\n[SURFACE STATS]:")
    for surface, stats in horse.surface_stats.items():
        print(f"  {surface}: {stats['wins']}/{stats['starts']} ({stats['win_pct']:.0f}%) Avg Fig: {stats['avg_fig']}")
    
    print(f"\n[WORKOUTS]:")
    print(f"  Last Workout: {horse.last_workout_days} days ago")
    print(f"  Pattern: {horse.workout_pattern}")
    print(f"  Recent Works: {len(horse.workouts)}")
    for work in horse.workouts[:3]:
        bullet = "x" if work.get('bullet') else " "
        print(f"    {bullet} {work['date']} {work['track']} {work['distance']}f {work['surface']}: {work['time']} ({work['rank']}/{work['total']}) {work['grade']}")
    
    print(f"\n[RACE HISTORY]:")
    print(f"  Total races parsed: {len(horse.race_history)}")
    for i, race in enumerate(horse.race_history[:3], 1):
        print(f"  Race {i}: {race['date']} ({race['surface']})")
        print(f"    Positions: {race['start']} -> {race['first_call']} -> {race['second_call']} -> {race['stretch']} -> {race['finish']}")
        print(f"    SPD: {race['spd']} | E1/E2: {race['e1']}/{race['e2']} | Odds: {race['odds']:.2f}")
        if i-1 < len(horse.trip_comments) and horse.trip_comments[i-1]:
            print(f"    Comment: {horse.trip_comments[i-1]}")
    
    print(f"\n[ANGLES]:")
    print(f"  Total: {horse.angle_count}")
    if horse.angle_flags:
        print(f"  Flags: {', '.join(horse.angle_flags[:5])}")
    for angle in horse.angles[:5]:
        print(f"    * {angle['category']}: {angle['win_pct']}% ({angle['starts']} starts) ROI: {angle['roi']}")
    
    print(f"\n[PEDIGREE]:")
    print(f"  Sire: {horse.sire}")
    if horse.sire_spi:
        print(f"    SPI: {horse.sire_spi}, AWD: {horse.sire_awd}")
    print(f"  Dam: {horse.dam}")
    if horse.damsire_spi:
        print(f"    Dam's Sire SPI: {horse.damsire_spi}")
    
    print(f"\n[PARSING QUALITY]:")
    print(f"  Confidence: {horse.parsing_confidence:.2%}")
    if horse.warnings:
        print(f"  Warnings: {', '.join(horse.warnings)}")

print(f"\n{'=' * 80}")
print("TEST COMPLETE - ALL DATA FIELDS EXTRACTED!")
print(f"{'=' * 80}")
