"""Quick diagnostic: Run parser on CT R6 PP text and show what was extracted."""

import sys

sys.path.insert(0, ".")
from elite_parser_v2_gold import GoldStandardBRISNETParser

# Read the PP text from SAMPLE_TEST_RACE.txt
with open("SAMPLE_TEST_RACE.txt", encoding="utf-8", errors="replace") as f:
    pp_text = f.read()

parser = GoldStandardBRISNETParser()
horses = parser.parse_full_pp(pp_text, debug=True)

print(f"\n{'=' * 80}")
print(f"PARSER RESULTS: {len(horses)} horses parsed")
print(f"{'=' * 80}")

for name, h in horses.items():
    print(f"\n{'─' * 60}")
    print(f"POST {h.post} | {name} ({h.pace_style} {h.quirin_points})")
    print(f"  Speed Figs: {h.speed_figures}")
    print(f"  Avg Top 2:  {h.avg_top2}")
    print(f"  Peak Fig:   {h.peak_fig}")
    print(f"  Last Fig:   {h.last_fig}")
    print(f"  Finishes:   {h.recent_finishes}")
    print(f"  Days Since: {h.days_since_last}")
    print(f"  Purses:     {h.recent_purses}")
    print(f"  Race Types: {h.race_types}")
    print(f"  Avg Purse:  {h.avg_purse}")
    print(f"  Prime Power:{h.prime_power}")
    print(f"  Trainer:    {h.trainer} ({h.trainer_win_pct}%)")
    print(f"  Jockey:     {h.jockey} ({h.jockey_win_pct}%)")
    print(f"  ML Odds:    {h.ml_odds}")
    print(f"  Confidence: {h.parsing_confidence:.1%}")
    print(f"  Errors:     {h.errors}")
    # Show first 200 chars of block for debugging
    block_preview = h.raw_block[:200].replace("\n", " | ")
    print(f"  Block[0:200]: {block_preview}")
