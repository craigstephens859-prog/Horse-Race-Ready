from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

DEFAULTS: Dict[str, Any] = {"exotic_bias_weights": {"runstyle": 1.0, "post": 1.0}}


def _coerce_num(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace("%", "")
        return float(s) if s else default
    except (ValueError, TypeError, AttributeError):
        return default


def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Returns the first existing column from candidates list.
    Case-insensitive, trimmed, allows symbols like '%', '#'.
    """
    if df is None or df.empty:
        return None
    
    # Normalize actual column names (strip whitespace, case-insensitive)
    actual_cols = {col.strip(): col for col in df.columns}
    actual_cols_lower = {col.lower(): orig for col, orig in actual_cols.items()}
    
    for candidate in candidates:
        # Try exact match first (after strip)
        candidate_stripped = candidate.strip()
        if candidate_stripped in actual_cols:
            return actual_cols[candidate_stripped]
        
        # Try case-insensitive match
        candidate_lower = candidate_stripped.lower()
        if candidate_lower in actual_cols_lower:
            return actual_cols_lower[candidate_lower]
    
    return None


def resolve_angle_columns(
    df: pd.DataFrame, user_map: Optional[Dict[str, str]] = None
) -> Dict[str, Optional[str]]:
    """
    Returns a dict of canonical->actual column mappings.
    user_map overrides auto-detection if provided.
    """
    # Define synonym sets for each canonical field
    synonym_sets = {
        "RunStyle": ["RunStyle", "Style", "Run Style", "RS"],
        "Post": ["Post", "#", "PP", "Gate", "Post Position"],
        "Quirin": ["Quirin", "QuirinPts", "Quirin Points", "Q"],
        "LastFig": [
            "LastFig",
            "Last BRIS Speed",
            "LastSpeed",
            "Last_Spd",
            "BRIS_SPD_L",
            "Last Spd",
        ],
        "SireROI": ["SireROI", "Sire ROI", "Sire_ROI", "SireROI%", "Sire ROI %"],
        "TrainerWin%": [
            "TrainerWin%",
            "Trainer Win %",
            "TrnWin%",
            "Trainer%",
            "Trainer_Win%",
        ],
        "JockeyWin%": [
            "JockeyWin%",
            "Jockey Win %",
            "JkyWin%",
            "Jockey%",
            "Jockey_Win%",
        ],
        "Workouts": ["Workouts", "Works", "Recent Works", "WorkoutNotes", "WO"],
        "DaysSinceLast": ["DaysSinceLast", "DaysSince", "DSR", "Days Since Last"],
    }
    
    resolved = {}
    
    for canonical, candidates in synonym_sets.items():
        # Check if user provided an override
        if user_map and canonical in user_map:
            resolved[canonical] = user_map[canonical]
        else:
            # Auto-detect from candidates
            resolved[canonical] = resolve_col(df, candidates)
    
    return resolved


def compute_eight_angles(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    debug: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes 8 normalized angle scores for handicapping.
    
    Args:
        df: DataFrame with horse data
        column_map: Optional dict to override column mappings (canonical -> actual)
        debug: If True, returns (angles_df, mapping_df) tuple
    
    Returns:
        angles_df if debug=False, else (angles_df, mapping_df)
    """
    if df is None or df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "Angle_EarlySpeed",
                "Angle_Class",
                "Angle_Pedigree",
                "Angle_Connections",
                "Angle_Post",
                "Angle_RunstyleBias",
                "Angle_WorkPattern",
                "Angle_Recency",
                "Angle_SourceCount",
                "Angles_Total",
            ]
        )
        if debug:
            mapping_df = pd.DataFrame(
                columns=["Canonical", "Resolved", "Status"]
            )
            return empty_df, mapping_df
        return empty_df
    
    # Resolve column mappings
    resolved_cols = resolve_angle_columns(df, column_map)
    
    # Build mapping dataframe for debugging
    mapping_rows = []
    for canonical, actual in resolved_cols.items():
        status = "FOUND" if actual is not None else "NOT FOUND"
        mapping_rows.append({
            "Canonical": canonical,
            "Resolved": actual if actual else "(none)",
            "Status": status
        })
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Helper to safely get column value
    def safe_get_col(canonical_name: str) -> pd.Series:
        actual_col = resolved_cols.get(canonical_name)
        if actual_col and actual_col in df.columns:
            return df[actual_col]
        # Return None series if column not found
        return pd.Series([None] * len(df), index=df.index)
    
    # Build angles dataframe
    out = pd.DataFrame(index=df.index)
    
    # Track which features were available per row
    source_counts = pd.Series([0] * len(df), index=df.index)
    
    # Angle 1: Early Speed (from last speed figure)
    last_fig_series = safe_get_col("LastFig")
    out["Angle_EarlySpeed"] = last_fig_series.apply(_coerce_num)
    source_counts += (last_fig_series.notna() & (last_fig_series != "")).astype(int)
    
    # Angle 2: Class (from Quirin points)
    quirin_series = safe_get_col("Quirin")
    out["Angle_Class"] = quirin_series.apply(_coerce_num)
    source_counts += (quirin_series.notna() & (quirin_series != "")).astype(int)
    
    # Angle 3: Pedigree (from Sire ROI)
    sire_roi_series = safe_get_col("SireROI")
    out["Angle_Pedigree"] = sire_roi_series.apply(_coerce_num)
    source_counts += (sire_roi_series.notna() & (sire_roi_series != "")).astype(int)
    
    # Angle 4: Connections (average of trainer and jockey win %)
    trainer_series = safe_get_col("TrainerWin%")
    jockey_series = safe_get_col("JockeyWin%")
    out["Angle_Connections"] = (
        trainer_series.apply(_coerce_num) + jockey_series.apply(_coerce_num)
    ) / 2.0
    source_counts += (trainer_series.notna() & (trainer_series != "")).astype(int)
    source_counts += (jockey_series.notna() & (jockey_series != "")).astype(int)
    
    # Angle 5: Post position
    post_series = safe_get_col("Post")
    out["Angle_Post"] = post_series.apply(_coerce_num)
    source_counts += (post_series.notna() & (post_series != "")).astype(int)
    
    # Angle 6: Run style bias
    runstyle_series = safe_get_col("RunStyle")
    rs = runstyle_series.astype(str).str.upper().fillna("")
    out["Angle_RunstyleBias"] = rs.map({"E": 3, "EP": 2, "E/P": 2, "P": 1, "S": 0}).fillna(0)
    source_counts += (runstyle_series.notna() & (runstyle_series != "")).astype(int)
    
    # Angle 7: Workout pattern (number of recent works)
    workouts_series = safe_get_col("Workouts")
    out["Angle_WorkPattern"] = workouts_series.astype(str).apply(
        lambda s: len([t for t in s.split() if t])
    )
    source_counts += (workouts_series.notna() & (workouts_series != "")).astype(int)
    
    # Angle 8: Recency (days since last race)
    days_series = safe_get_col("DaysSinceLast")
    out["Angle_Recency"] = (
        days_series.apply(lambda d: max(0.0, 60.0 - _coerce_num(d))).fillna(0)
    )
    source_counts += (days_series.notna() & (days_series != "")).astype(int)
    
    # Normalize all angles to 0-1 scale
    def _norm(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0
    
    cols = [
        "Angle_EarlySpeed",
        "Angle_Class",
        "Angle_Pedigree",
        "Angle_Connections",
        "Angle_Post",
        "Angle_RunstyleBias",
        "Angle_WorkPattern",
        "Angle_Recency",
    ]
    for c in cols:
        out[c] = _norm(out[c])
    
    # Add source count and total score
    out["Angle_SourceCount"] = source_counts
    out["Angles_Total"] = out[cols].sum(axis=1)
    
    if debug:
        return out, mapping_df
    return out


def apply_angles_to_ratings(ratings, angles: pd.DataFrame, weight: float = 0.25):
    if angles is None or angles.empty:
        return ratings
    bump = angles.get("Angles_Total", 0).astype(float)
    if isinstance(ratings, pd.DataFrame):
        base = ratings.get("R", None)
        if base is None:
            return ratings
        ratings["R_plus_angles"] = base.astype(float) * (1.0 - weight) + bump * weight
        return ratings
    return ratings.astype(float) * (1.0 - weight) + bump * weight
