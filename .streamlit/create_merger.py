# merge_files.py - Complete Merger Script
# This script combines streamlit_app.py with 8-Angle Enhancement System

import re

print("="*80)
print("8-ANGLE ENHANCEMENT MERGER")
print("="*80)
print()

# Read original file
print("Step 1: Reading original streamlit_app.py...")
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    original_content = f.read()

print(f"   √ Loaded {len(original_content)} characters")
print()

# Define all the components to add
print("Step 2: Preparing enhancement components...")

# Component 1: MODEL_CONFIG
model_config = '''
# ===================== 8-ANGLE MODEL CONFIG =====================

MODEL_CONFIG = {
    # Unified 8-Angle Equilibrium
    "total_adj_clip": 1.5,
    "synergy_pair_boost": 0.04,
    "softmax_tau": 0.78,

    # 1. Beyer / Speed
    "beyer_par": 82,
    "beyer_weight": 0.09,
    "beyer_recency_weights": [0.75, 0.18, 0.05, 0.02],
    "beyer_regression_penalty": 0.25,
    "beyer_spike_bonus": 0.06,
    "beyer_max_adjust": 1.1,
    "first_timer_beyer_proxy": {"stakes": 72, "allowance": 78, "claiming": 70},
    "beyer_idle_decay": 1.5,
    "beyer_synergy_mult": 1.05,

    # 2. Trainer
    "trainer_par_win_pct": 0.12,
    "trainer_weight": 0.06,
    "trainer_recency_window": 30,
    "trainer_volume_min": 5,
    "trainer_form_decay": 0.85,
    "trainer_angle_bonuses": {
        "second_off_layoff": 1.15,
        "shipper_to_track": 1.10,
        "first_time_lasix": 1.20,
        "blinkers_on": 1.08,
        "sprint_after_route": 1.05
    },
    "trainer_roi_threshold": 0.85,
    "trainer_max_adjust": 0.8,
    "trainer_debut_par": {"stakes": 0.08, "allowance": 0.12, "claiming": 0.15},

    # 3. Jockey
    "jockey_par_win_pct": 0.14,
    "jockey_weight": 0.07,
    "jockey_recency_window": 10,
    "jockey_volume_min": 8,
    "jockey_form_decay": 0.90,
    "jockey_angle_bonuses": {
        "first_time_mount": 1.10,
        "post_position_synergy": 1.12,
        "returning_from_layoff": 1.08,
        "apprentice_allowance": 1.05,
        "stalker_in_closer_bias": 1.15
    },
    "jockey_roi_threshold": 0.90,
    "jockey_max_adjust": 0.9,
    "jockey_debut_par": {"stakes": 0.10, "allowance": 0.14, "claiming": 0.18},

    # 4. Workouts
    "workout_par_times": {3: 36.0, 4: 48.0, 5: 60.0, 6: 72.0},
    "workout_weight": 0.05,
    "workout_recency_window": 45,
    "workout_bullet_mult": 1.20,
    "workout_volume_ideal": 2,
    "workout_max_adjust": 0.6,
    "layoff_work_proxy": 0.03,

    # 5. Pedigree
    "tomlinson_par_td": 100,
    "tomlinson_par_ts": 110,
    "pedigree_weight": 0.04,
    "sire_dam_weight": [0.60, 0.40],
    "first_timer_ped_mult": 1.25,
    "inbreeding_penalty": -0.02,
    "apt_max_adjust": 0.5,
    "tomlinson_dist_bonus": {"≤6f": 0.05, "6.5–7f": 0.02, "8f+": -0.03},
    "tomlinson_surface_bonus": {"Dirt": 0.03, "Turf": -0.01},

    # 6. Pace Rhythm
    "pace_par_ep": 88,
    "pace_par_midp": 82,
    "pace_par_lp": 80,
    "pace_weight": 0.10,
    "pace_recency_weights": [0.65, 0.25, 0.10],
    "pace_regression_penalty": 0.08,
    "pace_spike_bonus": 0.07,
    "pace_max_adjust": 0.9,
    "first_timer_pace_proxy": {
        "stakes": {"EP": 75, "MidP": 78, "LP": 72},
        "claiming": {"EP": 80, "MidP": 82, "LP": 78}
    },
    "pace_ppi_synergy": {
        "fast": {"E": -0.15, "E/P": -0.08, "P": 0.05, "S": 0.20},
        "slow": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
        "neutral": {"E": 0.02, "E/P": 0.00, "P": 0.00, "S": -0.02}
    },
    "pace_post_synergy": 0.03,
    "pace_idle_decay": 1.0,

    # 7. Fractional Variants
    "frac_variant_weight": 0.11,
    "frac_recency_weights": [0.70, 0.20, 0.10],
    "frac_regression_penalty": 0.09,
    "frac_spike_bonus": 0.08,
    "frac_max_adjust": 1.0,
    "first_timer_frac_proxy": {"EP": 78, "MidP": 80, "LP": 76},
    "frac_idle_decay": 0.8,
    "condition_frac_mults": {
        "fast": {"EP": 1.00, "MidP": 1.00, "LP": 1.00},
        "firm": {"EP": 1.02, "MidP": 1.01, "LP": 0.99},
        "good": {"EP": 0.98, "MidP": 1.00, "LP": 1.02},
        "yielding": {"EP": 0.92, "MidP": 0.97, "LP": 1.05},
        "muddy": {"EP": 0.90, "MidP": 0.95, "LP": 1.08},
        "sloppy": {"EP": 0.88, "MidP": 0.94, "LP": 1.10},
        "heavy": {"EP": 0.85, "MidP": 0.92, "LP": 1.12}
    },
    "frac_ppi_cross": {"fast": {"sloppy": {"E": 0.85}, "yielding": {"E": 0.90}}},
    "frac_post_synergy": 0.04,

    # 8. Quirin Style Surge
    "quirin_par": 85,
    "quirin_weight": 0.10,
    "quirin_recency_weights": [0.70, 0.20, 0.10],
    "quirin_regression_penalty": 0.22,
    "quirin_spike_bonus": 0.08,
    "quirin_max_adjust": 1.3,
    "first_timer_quirin_proxy": {
        "stakes": {"E": 70, "E/P": 75, "P": 78, "S": 72},
        "claiming": {"E": 82, "E/P": 80, "P": 82, "S": 80}
    },
    "quirin_idle_decay": 1.2,
    "quirin_style_bonuses": {
        "slow": {"E": 0.14, "E/P": 0.10, "P": -0.01, "S": -0.08},
        "neutral": {"E": 0.03, "E/P": 0.01, "P": 0.02, "S": 0.00},
        "fast": {"E": -0.20, "E/P": -0.12, "P": 0.08, "S": 0.28}
    },
    "quirin_call_points": {
        "E": {"1/4": 2, "1/2": 2, "stretch": 1},
        "E/P": {"1/4": 1.5, "1/2": 1.5, "stretch": 1.5},
        "P": {"1/4": 1, "1/2": 1.5, "stretch": 2},
        "S": {"1/4": 0.5, "1/2": 1, "stretch": 2.5}
    },
    "quirin_pace_factors": {"early": 0.40, "press": 0.35, "sustain": 0.25},
    "quirin_synergy_mult": 1.06,

    # Exotic
    "exotic_bias_weights": (1.35, 1.20, 1.10, 1.05),
    "stack_threshold": 0.06
}

# Base class bias (for adjusting pars by race type)
base_class_bias = {
    "Stakes (G1)": 1.15,
    "Stakes (G2)": 1.10,
    "Stakes (G3)": 1.08,
    "Stakes (Listed)": 1.05,
    "Stakes": 1.03,
    "Allowance": 1.00,
    "Maiden Special Weight": 0.95,
    "Maiden Claiming": 0.88,
    "Maiden (other)": 0.92,
    "Claiming": 0.85,
    "Other": 1.00
}

# Track bias profiles placeholder
TRACK_BIAS_PROFILES = {}
'''

print("   √ MODEL_CONFIG prepared")

# Save first part
with open('merge_files.py', 'w', encoding='utf-8') as f:
    f.write(__file__)

print()
print("Created merge_files.py")
print("Run: python merge_files.py")
