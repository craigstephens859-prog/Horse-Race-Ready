"""Test improved parser with actual BRISNET format"""

from elite_parser import EliteBRISNETParser
import re

# Sample BRISNET past performance text from Brown Sugar (Horse #1)
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
Sire Stats: AWD 6.8 17%Mud 1420MudSts 1.69spi
Dam'sSire: AWD 6.8 11%Mud 596MudSts 0.86spi
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
13Dec25GP¯ à 1ˆ fm :23¨ :46ª1:10¨ 1:40 ¨¨¬ ™TrPOaksL 125k¨¨¬ 83 87/ 84 +11 +7 86 12 11ƒ 12®ƒ 11¬ 9¬ 6«ƒ HusbandsMJ¨¨¯ 16.30 DstndOr²AndOnMorTm¨ƒRmsyPnd¨‚ No threat; 2-3wd 12
19Aug25Prx¨¨ 1Ñ ft :23¨ :46ª1:10ª 1:40ª ¨¨¬ ™CathSphiaL 200k¨¨ª 59 51/ 99 -7 0 72 10 13° 13¨© 12©© 12¨ 12¨° OrtizJL¨©¨ 5.10 DryPwdr«‚Ordaydramingirl¨ƒDiscoRles Always far back 13
01Jly25Ind« 1m gd :23© :46ª1:12 1:37ª ¦ ¨¨¨ ™A34000n2x ¨¨¬ 64 74/ 90 -3 -6 80 7 5«ƒ 4® 1¨ 1¨ 1ƒ PedrozaM¨¨° L *1.10 BrwnSgarƒFgtvStar¬MisSsanBª‚ Stmb;bmp;rail;rally3/8 8
29May25Ind¬ š 1ˆ ft :25 :50¨1:14« 1:46¨ ¦ ¨§° ™Mdn 32k ¨¨ª 65 68/ 97 -18 -14 83 5 2¨ 2 2 1¨ 1ª PedrozaM¨©§ L *0.05 BrwnSgrªThtrx¬Altlbtnaghty Press;bid5/16;mild drv 5
12Apr25GP à 1ˆ fm :23 :46ª1:11© 1:40« ¦ ¨¨¨ ™Mdn 55k ¨¨© 75 69/ 99 +1 -10 84 3 7ªƒ 8¬ 8ª‚ 3¨ 3¨ BravoJ¨¨¯ L *1.70 RyltyRt²RstlsDramr¨ BrnSgr³ 5p2nd;bid1/4;late gain 8
26Jan25GP¨¨ à 1m fm :23© :47«1:11ª 1:34© ¨¨¨ ™Mdn 89k ¨¨© 83 78/ 82 -1 -5 77 2 1 1 1 2² 2« SaezL¨¨¯ Lbf *1.90 ClscQ«BrwnSgr¨ƒChpgnBrnch¨‚ Ins;roused1/4;held plc 12
"""

parser = EliteBRISNETParser()

print("=" * 80)
print("TESTING PARSER IMPROVEMENTS WITH ACTUAL BRISNET FORMAT")
print("=" * 80)

# Test speed figure extraction
print("\n1. SPEED FIGURE PATTERN TEST:")
print("-" * 60)
speed_pattern = parser.SPEED_FIG_PATTERN
matches = list(speed_pattern.finditer(test_pp))
print(f"Found {len(matches)} speed figure matches")
for i, match in enumerate(matches[:5], 1):
    print(f"  Match {i}: Date={match.group(1)}, SPD={match.group(2)}")

# Test race type/purse extraction
print("\n2. RACE TYPE & PURSE PATTERN TEST:")
print("-" * 60)
purse_pattern = r'™([A-Za-z][A-Za-z0-9\s]*?)\s*(\d+)k?(?:[¨°©ª¬«­®¯±²³´]|\s)'
matches = list(re.finditer(purse_pattern, test_pp))
print(f"Found {len(matches)} race type/purse matches")
for i, match in enumerate(matches, 1):
    race_info = match.group(1).strip()
    purse_str = match.group(2)
    print(f"  Match {i}: RaceInfo='{race_info}', Purse={purse_str}")

# Test finish position extraction
print("\n3. FINISH POSITION PATTERN TEST:")
print("-" * 60)
finish_pattern = r'(\d{2}[A-Za-z]{3}\d{2}).*?FIN\s+(\d{1,2})[ƒ®«ª³©¨°¬²‚±\s]'
matches = list(re.finditer(finish_pattern, test_pp))
print(f"Found {len(matches)} finish position matches")
for i, match in enumerate(matches, 1):
    print(f"  Match {i}: Date={match.group(1)}, Finish={match.group(2)}")

# Test full parsing
print("\n4. FULL HORSE PARSING TEST:")
print("-" * 60)
horses = parser.parse_full_pp(test_pp)
if horses:
    for horse_name, horse in horses.items():
        print(f"\nHorse: {horse_name}")
        print(f"  Speed Figures: {horse.speed_figures}")
        print(f"  Avg Top 2: {horse.avg_top2}")
        print(f"  Peak Figure: {horse.peak_fig}")
        print(f"  Last Figure: {horse.last_fig}")
        print(f"  Recent Purses: {horse.recent_purses}")
        print(f"  Race Types: {horse.race_types}")
        print(f"  Recent Finishes: {horse.recent_finishes}")
        print(f"  Days Since Last: {horse.days_since_last}")
else:
    print("  ERROR: No horses parsed!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
