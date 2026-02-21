"""Debug script: Why Don't Get Cute and Just a Wizard get 0.00 ratings"""

import logging

logging.basicConfig(level=logging.WARNING)

import numpy as np

from elite_parser import GoldStandardBRISNETParser
from unified_rating_engine import UnifiedRatingEngine

# Read the PP text from session (simulate what the app does)
# We'll use a minimal test to check if parser finds these horses
pp_text = open("_ct_r6_pp.txt", encoding="utf-8", errors="replace").read()

# Check if this file actually has R5 data... if not, we'll know
print("=" * 60)
print("STEP 1: Parse with Elite Parser")
print("=" * 60)

parser = GoldStandardBRISNETParser()
horses = parser.parse_full_pp(pp_text, debug=False)

print(f"\nParsed {len(horses)} horses:")
for name, h in horses.items():
    figs = h.speed_figures if h.speed_figures else []
    print(
        f"  {name}: post={h.post}, figs={figs[:3]}, avg_top2={h.avg_top2}, "
        f"pace_style={h.pace_style}, parsing_conf={h.parsing_confidence:.2f}"
    )

# Check specific horses
target_horses = ["Don't Get Cute", "Dont Get Cute", "Just a Wizard"]
print("\nLooking for target horses:")
for t in target_horses:
    found = t in horses
    # Also try normalized match
    norm_t = " ".join(t.replace("'", "").replace("`", "").lower().split())
    norm_match = any(
        " ".join(n.replace("'", "").replace("`", "").lower().split()) == norm_t
        for n in horses.keys()
    )
    print(f"  '{t}': exact={found}, normalized={norm_match}")

print("\n" + "=" * 60)
print("STEP 2: Test Unified Rating Engine")
print("=" * 60)

engine = UnifiedRatingEngine(softmax_tau=3.0)
results_df = engine.predict_race(
    pp_text=pp_text,
    today_purse=34100,
    today_race_type="allowance",
    track_name="Charles Town",
    surface_type="Dirt",
    distance_txt="4.5f",
)

if not results_df.empty:
    print(f"\nEngine produced {len(results_df)} horses:")
    for _, row in results_df.iterrows():
        rating = row.get("Rating", "N/A")
        is_nan = isinstance(rating, float) and (
            np.isnan(rating) or not np.isfinite(rating)
        )
        print(
            f"  {row['Horse']}: Rating={rating} {'** NaN! **' if is_nan else ''}, "
            f"Cclass={row.get('Cclass', 'N/A')}, Cform={row.get('Cform', 'N/A')}, "
            f"Cspeed={row.get('Cspeed', 'N/A')}, Cpace={row.get('Cpace', 'N/A')}, "
            f"Cstyle={row.get('Cstyle', 'N/A')}, Cpost={row.get('Cpost', 'N/A')}"
        )
else:
    print("Engine returned EMPTY results!")

print("\nDone.")
