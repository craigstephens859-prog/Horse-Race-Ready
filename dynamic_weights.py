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

    # Surface adjustments — research shows turf is 15-25% less pace-dependent
    if "turf" in surf:
        base["class_form"] = (
            1.35  # Class far more predictive on turf (pedigree/class = king)
        )
        base["pace_speed"] = (
            0.75  # Pace much less dominant on turf (tactics matter more)
        )
        base["track_bias"] = 1.25  # Turf biases (rail/post) can be very strong
        base["trs_jky"] = 1.15  # Jockey ride crucial on turf
    elif "synth" in surf or "all-weather" in surf:
        base["class_form"] = 1.15
        base["pace_speed"] = 0.90
    else:  # Dirt
        base["pace_speed"] = (
            1.20  # Speed/pace dominant on dirt (Beyer figures predict 40%)
        )
        base["class_form"] = 1.0

    # Distance adjustments — sprints vs routes have fundamentally different dynamics
    if dist_bucket == "≤6f":  # Sprint
        base["pace_speed"] *= 1.30  # Pace critical in sprints (early speed wins 60%+)
        base["class_form"] *= 0.85  # Class less decisive in short bursts
        base["style_post"] *= 1.15  # Post position matters more in sprints
    elif dist_bucket == "6.5–7f":  # Middle distance — most competitive
        base["pace_speed"] *= 1.10
        base["class_form"] *= 1.10
    else:  # Route (8f+)
        base["pace_speed"] *= 0.80  # Pace scenarios less predictive in routes
        base["class_form"] *= 1.30  # Class/stamina separation is maximum in routes
        base["trs_jky"] *= 1.10  # Trainer conditioning for distance matters

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
        # Elite stakes — all entrants are quality, form cycle is the differentiator
        w["class_form"] = w.get("class_form", 1.0) * 1.40
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.80
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.25  # Top jockeys/trainers matter most
    elif "g3" in rt or "stakes" in rt:
        w["class_form"] = w.get("class_form", 1.0) * 1.25
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.88
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.15
    elif "allowance" in rt or "alw" in rt:
        # Balanced approach for allowance — slight class emphasis
        w["class_form"] = w.get("class_form", 1.0) * 1.10
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.05
    elif "claiming" in rt or "clm" in rt:
        # Claiming — recent form/speed key, class differences minimal
        w["class_form"] = w.get("class_form", 1.0) * 0.75
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.25
        w["track_bias"] = (
            w.get("track_bias", 1.0) * 1.15
        )  # Biases exploitable in cheap races
    elif "maiden" in rt:
        # Maiden — limited history, pedigree/trainer/workout angles matter
        w["class_form"] = w.get("class_form", 1.0) * 0.85
        w["track_bias"] = w.get("track_bias", 1.0) * 1.15
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.20  # Trainer debut patterns critical

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
        w["class_form"] = w.get("class_form", 1.0) * 1.30
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.85
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.25
    elif purse_val >= 200000:  # Stakes/listed
        w["class_form"] = w.get("class_form", 1.0) * 1.20
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.15
    elif purse_val >= 100000:  # Quality allowance
        w["class_form"] = w.get("class_form", 1.0) * 1.10
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.10
    elif purse_val >= 50000:  # Mid-level
        pass  # Use base weights
    elif purse_val >= 20000:  # Lower claiming
        w["class_form"] = w.get("class_form", 1.0) * 0.85
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.20
    else:  # Bottom level ($15k and below)
        w["class_form"] = w.get("class_form", 1.0) * 0.75
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.20
        w["track_bias"] = w.get("track_bias", 1.0) * 1.30

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
        # Off-track: pace scenarios massively disrupted, class/pedigree rises
        w["pace_speed"] = (
            w.get("pace_speed", 1.0) * 0.70
        )  # Pace nearly irrelevant in slop
        w["class_form"] = w.get("class_form", 1.0) * 1.25  # Better horses handle mud
        w["track_bias"] = (
            w.get("track_bias", 1.0) * 1.40
        )  # Rail/inside is critical in mud
        w["style_post"] = w.get("style_post", 1.0) * 1.15  # Post position crucial
    elif "good" in cond or "yield" in cond:
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.90
        w["class_form"] = w.get("class_form", 1.0) * 1.10
        w["track_bias"] = w.get("track_bias", 1.0) * 1.10
    # Fast/Firm = no adjustment

    return w
