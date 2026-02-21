# config.py
# Horse Race Ready — Configuration Constants & Data
# Extracted from app.py (Phase 1 module split) for modularity.
# This module contains ONLY pure data — no function definitions, no external imports.

# ===================== String Constants =====================
DIST_BUCKET_SPRINT = "≤6f"
DIST_BUCKET_MID = "6.5–7f"
DIST_BUCKET_ROUTE = "8f+"
COL_FAIR_PCT = "Fair %"
COL_FAIR_ODDS = "Fair Odds"
COL_EDGE_PP = "Edge (pp)"
COL_EV_PER_DOLLAR = "EV per $1"
COL_PROB_PCT = "Prob %"
DEFAULT_DISTANCE = "6 Furlongs"
BIAS_FAIR_NEUTRAL = "fair/neutral"
RACE_TYPE_MAIDEN_SP_WT = "maiden special weight"
RACE_TYPE_MAIDEN_CLM = "maiden claiming"

# ===================== First-Time Starter (FTS) Parameters =====================
# Maiden Special Weight (MSW) - Define FTS Parameters Dictionary
# Handles horses with zero previous racing history
FTS_PARAMS = {
    "jockey_confidence": 0.7,  # Boost for strong bookings
    "trainer_confidence": 0.9,  # Key for top trainers (>20% FTS win rate)
    "speed_confidence": 0.4,  # Low; use workouts if available
    "form_confidence": 0.0,  # Disabled, no prior races
    "class_confidence": 0.6,  # Implies quality but unproven
    "pedigree_confidence": 0.8,  # Crucial for debuts
    "ml_odds_confidence": 0.8,  # Market accuracy for elite FTS
    "live_odds_confidence": 0.9,  # Insider sentiment
    "base_multiplier": 0.82,  # Base FTS rating multiplier (82% of normal — was 75%, too harsh)
    "elite_trainer_multiplier": 1.15,  # Boost for elite connections (82% * 1.15 = 94%)
}

# Elite Trainers: Top 5% debut ROI (expand based on historical data)
ELITE_TRAINERS = {
    # Tier 1: Hall of Fame / perennial leaders (20%+ win rate, $10M+ earnings)
    "Todd Pletcher",
    "Bob Baffert",
    "Chad Brown",
    "Steve Asmussen",
    "Brad Cox",
    "Bill Mott",
    "Shug McGaughey",
    "Mark Casse",
    # Tier 2: Top 20 national trainers by wins/earnings
    "Wesley Ward",
    "Jack Sisterson",
    "Doug O'Neill",
    "Mike Maker",
    "Dale Romans",
    "Christophe Clement",
    "Graham Motion",
    "Ken McPeek",
    "Dallas Stewart",
    "Peter Miller",
    "John Sadler",
    "Michael McCarthy",
    "Phil D'Amato",
    "Jorge Navarro",
    "Linda Rice",
    # Tier 3: High-percentage specialists (may not lead wins but elite strike rate)
    "Tom Amoss",
    "Ian Wilkes",
    "Brendan Walsh",
    "Eddie Kenneally",
    "Jeff Bonde",
    "Saffie Joseph Jr.",
    "Danny Gargan",
    "John Kimmel",
    "Rudy Rodriguez",
    "Al Stall Jr.",
    "D. Wayne Lukas",
    "Larry Jones",
}  # Set for O(1) lookup performance

# ===================== NA Running Style + QSP Parameters =====================
# Horses with "NA" running style: BRISNET couldn't assign a style because the
# horse lacks sufficient starts at this exact distance/surface combination.
# NA ≠ bad — it means UNKNOWN. We use Quirin Speed Points (0-8) to infer
# partial early-speed tendency and adjust confidence/ratings accordingly.
NA_STYLE_PARAMS = {
    "form_confidence": 0.0,  # No meaningful form cycle at this dist/surface
    "speed_confidence": 0.5,  # Workouts/morning line provide some signal
    "pedigree_boost": 0.2,  # Pedigree becomes more important for unknowns
    "trainer_boost": 0.1,  # Trainer patterns matter more for unknowns
    "style_penalty": -0.15,  # Gentler than -0.3 (unknown ≠ mismatch)
    "rating_dampener": 0.85,  # 15% overall reduction for style uncertainty
    "fts_na_dampener": 0.92,  # Mild additional dampener when FTS+NA stack
    "qsp_speed_scaling": 0.3,  # Max speed_confidence boost from QSP: += (q/8)*0.3
    "qsp_ep_threshold": 5,  # QSP >= 5 infers partial E/P tendency
    "qsp_style_offset": 0.15,  # Max style penalty offset from high QSP
}

# ===================== Model Config & Tuning =====================


MODEL_CONFIG = {
    # --- Rating Model ---
    "softmax_tau": 3.0,  # Controls win prob "sharpness". Must match unified engine rating scale.
    # CALIBRATED Feb 5, 2026: Was 0.85 which created 96%+ concentration on single horse.
    # Unified engine raw Rating sums have 5-10pt spreads; tau=3.0 gives realistic odds.
    "speed_fig_weight": 0.15,  # OPTIMIZED Feb 9 2026: Was 0.05 (speed was irrelevant).
    # 0.15 = 10 fig points = 1.5 bonus. Consistent with Beyer/Quirin/Benter research
    # that speed figures predict ~30-40% of race outcomes.
    "first_timer_fig_default": 50,  # Assumed speed fig for a 1st-time starter.
    # --- Pace & Style Model ---
    "ppi_multiplier": 1.5,  # Overall impact of the Pace Pressure Index (PPI).
    "ppi_tailwind_factor": 0.6,  # How much of the PPI value is given to E/EP or S horses.
    "style_strength_weights": {  # Multiplier for pace tailwind based on strength.
        "Strong": 1.0,
        "Solid": 0.8,
        "Slight": 0.5,
        "Weak": 0.3,
    },
    # --- Manual Bias Model (Section B) ---
    "style_match_table": {
        "favoring": {"E": 0.85, "E/P": 0.65, "P": -0.25, "S": -0.60},
        "closer favoring": {"E": -0.60, "E/P": -0.25, "P": 0.35, "S": 0.65},
        "fair/neutral": {"E": 0.0, "E/P": 0.0, "P": 0.0, "S": 0.0},
    },
    "style_quirin_threshold": 6,  # Quirin score needed for "strong" style bonus.
    "style_quirin_bonus": 0.15,  # INCREASED from 0.10 — Quirin matters more
    "post_bias_rail_bonus": 0.50,
    "post_bias_inner_bonus": 0.35,
    "post_bias_mid_bonus": 0.35,
    "post_bias_outside_bonus": 0.35,
    # --- Pedigree & Angle Tweaks (Cclass) ---
    "ped_dist_bonus": 0.06,
    "ped_dist_penalty": -0.04,  # Note: This should be negative
    "ped_dist_neutral_bonus": 0.03,
    "ped_first_pct_threshold": 14,  # Sire/Damsire 1st-time-win %
    "ped_first_pct_bonus": 0.02,
    "angle_debut_msw_bonus": 0.05,
    "angle_debut_other_bonus": 0.03,
    "angle_debut_sprint_bonus": 0.01,
    "angle_second_career_bonus": 0.03,
    "angle_surface_switch_bonus": 0.02,
    "angle_blinkers_on_bonus": 0.02,
    "angle_blinkers_off_bonus": 0.005,
    "angle_shipper_bonus": 0.01,
    "angle_off_track_route_bonus": 0.01,
    "angle_roi_pos_max_bonus": 0.06,  # Max bonus from positive ROI angles
    "angle_roi_pos_per_bonus": 0.01,  # Bonus per positive ROI angle
    "angle_roi_neg_max_penalty": 0.03,  # Max penalty from negative ROI angles (applied as -)
    "angle_roi_neg_per_penalty": 0.005,  # Penalty per negative ROI angle (applied as -)
    "angle_tweak_min_clip": -0.12,  # Min/Max total adjustment from all angles
    "angle_tweak_max_clip": 0.12,
    # --- Exotics & Strategy ---
    "exotic_bias_weights": (
        1.30,
        1.15,
        1.05,
        1.03,
    ),  # (1st, 2nd, 3rd, 4th) Harville bias
    "strategy_confident": {  # Placeholders, not used by new strategy builder
        "ex_max": 4,
        "ex_min_prob": 0.020,
        "tri_max": 6,
        "tri_min_prob": 0.010,
        "sup_max": 8,
        "sup_min_prob": 0.008,
    },
    "strategy_value": {  # Placeholders, not used by new strategy builder
        "ex_max": 6,
        "ex_min_prob": 0.015,
        "tri_max": 10,
        "tri_min_prob": 0.008,
        "sup_max": 12,
        "sup_min_prob": 0.006,
    },
}

# =========================
# Track parsing, race-type, distance options, and track-bias integration
# =========================

# -------- Distance options (UI) --------
DISTANCE_OPTIONS = [
    # Short sprints
    "4 Furlongs",
    "4 1/2 Furlongs",
    "4.5 Furlongs",
    "5 Furlongs",
    "5 1/2 Furlongs",
    "5.5 Furlongs",
    "6 Furlongs",
    "6 1/2 Furlongs",
    "6.5 Furlongs",
    "7 Furlongs",
    "7 1/2 Furlongs",
    "7.5 Furlongs",
    # Routes & variants
    "1 Mile",
    "1 Mile 70 Yards",
    "1 1/16 Miles",
    "1 1/8 Miles",
    "1 3/16 Miles",
    "1 1/4 Miles",
    "1 5/16 Miles",
    "1 3/8 Miles",
    "1 7/16 Miles",
    "1 1/2 Miles",
    "1 9/16 Miles",
    "1 5/8 Miles",
    "1 3/4 Miles",
    "1 7/8 Miles",
    "2 Miles",
]

# -------- Canonical track names + aliases --------
TRACK_ALIASES = {
    # === Major / Grade 1 Tracks ===
    "Aqueduct": ["aqueduct", "aqu"],
    "Belmont Park": [
        "belmont park",
        "belmont at aqueduct",
        "aqueduct at belmont",
        "belmont at the big a",
        "big a",
        "belmont",
        "bel",
    ],
    "Churchill Downs": ["churchill downs", "churchill", "cd"],
    "Del Mar": ["del mar", "dmr"],
    "Gulfstream Park": ["gulfstream park", "gulfstream", "gp"],
    "Keeneland": ["keeneland", "kee"],
    "Laurel Park": ["laurel park", "laurel", "lrl"],
    "Oaklawn Park": ["oaklawn park", "oaklawn", "op"],
    "Pimlico": ["pimlico", "pim"],
    "Santa Anita": ["santa anita park", "santa anita", "sa"],
    "Saratoga": ["saratoga", "sar"],
    # === Secondary / Regional Tracks ===
    "Ak-Sar-Ben": ["ak-sar-ben", "aksarben", "akr"],
    "Albuquerque Downs": ["albuquerque downs", "albuquerque", "abq"],
    "Arapahoe Park": ["arapahoe park", "arapahoe", "arp"],
    "Arlington Park": ["arlington park", "arlington", "ap"],
    "Belterra Park": ["belterra park", "belterra", "btp"],
    "Canterbury Park": ["canterbury park", "canterbury", "cbp"],
    "Century Mile": ["century mile", "cym"],
    "Charles Town": ["charles town", "charlestown", "ct"],
    "Colonial Downs": ["colonial downs", "colonial", "cln"],
    "Columbus": ["columbus", "clb"],
    "Delaware Park": ["delaware park", "delaware", "del"],
    "Delta Downs": ["delta downs", "ded"],
    "Ellis Park": ["ellis park", "ellis", "elp"],
    "Emerald Downs": ["emerald downs", "emerald", "emr", "emd"],
    "Evangeline Downs": ["evangeline downs", "evangeline", "evd"],
    "Fair Grounds": ["fair grounds", "fairgrounds", "fg"],
    "Fair Meadows": ["fair meadows", "fmr"],
    "Fairmount Park": [
        "fairmount park",
        "fanduel fairmount",
        "cah",
        "collinsville",
        "fmp",
    ],
    "Finger Lakes": ["finger lakes", "fl"],
    "Fonner Park": ["fonner park", "fonner", "fon"],
    "Fort Erie": ["fort erie", "fe"],
    "Golden Gate Fields": ["golden gate fields", "golden gate", "ggf", "gg"],
    "Grants Pass": ["grants pass", "grp"],
    "Great Lakes Downs": ["great lakes downs", "great lakes", "gld"],
    "Gulfstream Park West": ["gulfstream park west", "gulfstream west", "gpw"],
    "Hastings": ["hastings", "hst"],
    "Hawthorne": ["hawthorne", "haw"],
    "Horseshoe Indianapolis": [
        "horseshoe indianapolis",
        "indiana grand",
        "ind",
        "indy",
        "hsi",
    ],
    "Kentucky Downs": ["kentucky downs", "kd"],
    "Lone Star Park": ["lone star park", "lone star", "ls"],
    "Los Alamitos": ["los alamitos", "lam"],
    "Louisiana Downs": ["louisiana downs", "lad"],
    "Mahoning Valley": ["mahoning valley", "mahoning", "mvr"],
    "Monmouth Park": ["monmouth park", "monmouth", "mth"],
    "Mountaineer": ["mountaineer", "mnr"],
    "Parx Racing": ["parx racing", "parx", "philadelphia park", "prx"],
    "Penn National": ["penn national", "penn", "pen"],
    "Pleasanton": ["pleasanton", "pln"],
    "Portland Meadows": ["portland meadows", "portland", "pm"],
    "Prairie Meadows": ["prairie meadows", "prairie", "prm"],
    "Presque Isle Downs": ["presque isle downs", "presque isle", "pid"],
    "Remington Park": ["remington park", "remington", "rp"],
    "Retama Park": ["retama park", "retama", "ret"],
    "Ruidoso Downs": ["ruidoso downs", "ruidoso", "rud"],
    "Sam Houston": ["sam houston race park", "sam houston", "hou"],
    "Santa Rosa": ["santa rosa", "sr"],
    "Sunland Park": ["sunland park", "sunland", "sun"],
    "Sunray Park": ["sunray park", "sunray", "sry"],
    "Suffolk Downs": ["suffolk downs", "suffolk", "suf"],
    "Tampa Bay Downs": ["tampa bay downs", "tampa bay", "tampa", "tam"],
    "Thistledown": ["thistledown", "tdn"],
    "Turfway Park": ["turfway park", "turfway", "tp"],
    "Turf Paradise": ["turf paradise", "tup"],
    "Will Rogers Downs": ["will rogers downs", "will rogers", "wrd"],
    "Woodbine": ["woodbine", "wo"],
    "Zia Park": ["zia park", "zia", "zp"],
}
_CANON_BY_TOKEN: dict[str, str] = {}
for _canon, _toks in TRACK_ALIASES.items():
    for _t in _toks:
        _CANON_BY_TOKEN[_t] = _canon

# -------- Race-type constants + detection --------
# This dictionary is our constant. It measures the "reliability" of the race type.
base_class_bias = {
    "stakes (g1)": 0.90,  # PEGASUS TUNING: G1 class weight reduced in rating engine (10.0→6.0)
    "stakes (g2)": 0.92,
    "stakes (g3)": 0.93,
    "stakes (listed)": 0.95,
    "stakes": 0.95,
    "allowance optional claiming (aoc)": 0.96,
    "maiden special weight": 0.97,
    "allowance": 0.99,
    "starter handicap": 1.02,
    "starter allowance": 1.03,
    "starter optional claiming": 1.00,
    "waiver claiming": 1.07,
    "claiming": 1.12,
    "maiden claiming": 1.15,
}

condition_modifiers = {
    "fast": 1.0,
    "firm": 1.0,
    "good": 1.03,
    "yielding": 1.04,
    "muddy": 1.08,
    "sloppy": 1.10,
    "heavy": 1.10,
}

# -------- Track bias profiles (additive deltas; conservative magnitude) --------
TRACK_BIAS_PROFILES = {
    "Keeneland": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.35, "E/P": 0.20, "P": -0.10, "S": -0.25},
                "post": {"rail": 0.20, "inner": 0.10, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Del Mar": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Churchill Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Kentucky Downs": {
        "Turf": {
            "≤6f": {
                "runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "8f+": {
                "runstyle": {"E": -0.10, "E/P": 0.00, "P": 0.10, "S": 0.20},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
        }
    },
    "Saratoga": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Santa Anita": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Mountaineer": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        }
    },
    "Charles Town": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.45, "E/P": 0.25, "P": -0.15, "S": -0.35},
                "post": {"rail": 0.25, "inner": 0.15, "mid": -0.05, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.30, "E/P": 0.20, "P": -0.10, "S": -0.25},
                "post": {"rail": 0.15, "inner": 0.10, "mid": -0.05, "outside": -0.10},
            },
            "8f+": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Gulfstream Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
    },
    "Tampa Bay Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Belmont Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Horseshoe Indianapolis": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        },
    },
    "Penn National": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        }
    },
    "Presque Isle Downs": {
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
        }
    },
    "Woodbine": {
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Evangeline Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Oaklawn Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.30, "E/P": 0.15, "P": -0.10, "S": -0.20},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Fair Grounds": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.20},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Aqueduct": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Laurel Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Fairmount Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Finger Lakes": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    # Default fallback profile for tracks not specifically listed
    "_DEFAULT": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
            "8f+": {
                "runstyle": {"E": 0.03, "E/P": 0.03, "P": 0.03, "S": -0.03},
                "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.08},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.03, "P": 0.05, "S": -0.03},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03},
            },
        },
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.03, "E/P": 0.05, "P": 0.05, "S": -0.03},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": 0.00},
            },
        },
    },
}
