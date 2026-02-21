"""
dynamic_weights.py - Phase 5 extraction from app.py
Dynamic weight presets and adjustment pipeline for race handicapping.
"""

from utils import distance_bucket




def get_weight_preset(surface: str, distance: str) -> dict:
    """
    Generate base weights based on surface type and distance.

    Surface Logic:
    - Dirt: Speed figures and pace more predictive
    - Turf: Class/form and pedigree more predictive
    - Synthetic: Balanced approach

    Distance Logic:
    - Sprint (≤7f): Pace/speed dominates
    - Route (≥8f): Class/stamina more important
    """
    surf = (surface or "Dirt").strip().lower()
    dist_bucket = distance_bucket(distance) if distance else "8f+"

    base = {
        "class_form": 1.0,
        "pace_speed": 1.0,
        "style_post": 1.0,
        "track_bias": 1.0,
        "trs_jky": 1.0,
    }

    # Surface adjustments
    if "turf" in surf:
        base["class_form"] = 1.25  # Class more predictive on turf
        base["pace_speed"] = 0.85  # Pace less dominant on turf
        base["track_bias"] = 1.15  # Turf biases can be strong
    elif "synth" in surf or "all-weather" in surf:
        base["class_form"] = 1.10
        base["pace_speed"] = 0.95
    else:  # Dirt
        base["pace_speed"] = 1.15  # Speed/pace more dominant on dirt
        base["class_form"] = 1.0

    # Distance adjustments
    if dist_bucket == "≤6f":  # Sprint
        base["pace_speed"] *= 1.20  # Pace critical in sprints
        base["class_form"] *= 0.90  # Class less predictive in sprints
    elif dist_bucket == "6.5–7f":  # Middle distance
        base["pace_speed"] *= 1.05
        base["class_form"] *= 1.05
    else:  # Route (8f+)
        base["pace_speed"] *= 0.85  # Pace less dominant in routes
        base["class_form"] *= 1.20  # Class/stamina key in routes

    return base




def apply_strategy_profile_to_weights(weights: dict, profile: str) -> dict:
    """
    Adjust weights based on user's strategy profile.

    Confident: Favor top-rated horses (class/form emphasis)
    Value Hunter: Look for overlays (pace/track bias emphasis)
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}

    w = weights.copy()
    profile_lower = (profile or "").lower()

    if "value" in profile_lower:
        # Value hunters look for pace/bias edges
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.20
        w["class_form"] = w.get("class_form", 1.0) * 0.90
    else:  # Confident
        # Confident players trust class/form
        w["class_form"] = w.get("class_form", 1.0) * 1.15
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.95

    return w




def adjust_by_race_type(weights: dict, race_type: str) -> dict:
    """
    Adjust weights based on race type/class.

    Stakes (G1-G3): Class separation critical, pace less dominant
    Allowance: Balanced
    Claiming: Form/recent races more important than class
    Maiden: Pedigree/debut angles more important
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}

    w = weights.copy()
    rt = (race_type or "").lower()

    if "g1" in rt or "g2" in rt:
        # Elite stakes - class differences narrow, form is key
        w["class_form"] = w.get("class_form", 1.0) * 1.30
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.85
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.15  # Top jockeys matter
    elif "g3" in rt or "stakes" in rt:
        w["class_form"] = w.get("class_form", 1.0) * 1.20
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.90
    elif "allowance" in rt:
        # Balanced approach for allowance
        w["class_form"] = w.get("class_form", 1.0) * 1.05
    elif "claiming" in rt or "clm" in rt:
        # Claiming - form/recent performance key, class less predictive
        w["class_form"] = w.get("class_form", 1.0) * 0.80
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
    elif "maiden" in rt:
        # Maiden - limited data, pedigree/angles matter
        w["class_form"] = w.get("class_form", 1.0) * 0.90
        w["track_bias"] = w.get("track_bias", 1.0) * 1.10

    return w




def apply_purse_scaling(weights: dict, purse: int) -> dict:
    """
    Scale weights based on purse amount (proxy for overall race quality).

    Higher purse = more reliable data, tighter competition
    Lower purse = more variance, pace/bias edges more exploitable
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}

    w = weights.copy()
    purse_val = purse or 0

    if purse_val >= 500000:  # Major stakes ($500k+)
        w["class_form"] = w.get("class_form", 1.0) * 1.25
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.90
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.20
    elif purse_val >= 100000:  # Quality stakes/allowance
        w["class_form"] = w.get("class_form", 1.0) * 1.10
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.10
    elif purse_val >= 50000:  # Mid-level
        pass  # Use base weights
    elif purse_val >= 20000:  # Lower claiming
        w["class_form"] = w.get("class_form", 1.0) * 0.90
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.10
        w["track_bias"] = w.get("track_bias", 1.0) * 1.15
    else:  # Bottom level
        w["class_form"] = w.get("class_form", 1.0) * 0.80
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.20

    return w




def apply_condition_adjustment(weights: dict, condition: str) -> dict:
    """
    Adjust weights based on track condition.

    Fast/Firm: Standard weights
    Good/Yielding: Stamina/class slightly more important
    Muddy/Sloppy/Heavy: Off-track specialists, pace less predictive
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}

    w = weights.copy()
    cond = (condition or "fast").lower()

    if "mud" in cond or "slop" in cond or "heavy" in cond:
        # Off-track - pace scenarios disrupted
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.80
        w["class_form"] = w.get("class_form", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.30  # Rail/post matters more
    elif "good" in cond or "yield" in cond:
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.95
        w["class_form"] = w.get("class_form", 1.0) * 1.05
    # Fast/Firm = no adjustment

    return w

