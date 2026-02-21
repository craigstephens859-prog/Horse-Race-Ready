"""
strategy_builder.py - Phase 5 extraction from app.py
Betting strategy builder and component breakdown for race analysis.
"""

import logging
import re

import numpy as np
import pandas as pd
import streamlit as st

try:
    from race_class_parser import parse_and_calculate_class

    RACE_CLASS_PARSER_AVAILABLE = True
except ImportError:
    RACE_CLASS_PARSER_AVAILABLE = False
    parse_and_calculate_class = None

logger = logging.getLogger(__name__)


def str_to_decimal_odds(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        # CHECK AMERICAN ODDS FIRST (must precede generic numeric check)
        # American: "+250" → (250/100)+1 = 3.5, "-150" → (100/150)+1 = 1.667
        if re.fullmatch(r"\+\d+", s):
            return 1.0 + float(s) / 100.0
        if re.fullmatch(r"-\d+", s):
            return 1.0 + 100.0 / abs(float(s))
        # Fractional: "5/2" → 5/2+1 = 3.5, "3-1" → 3/1+1 = 4.0
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) + 1.0
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a) / float(b) + 1.0
        # Decimal odds or plain number (e.g. "3.5", "2.10")
        if re.fullmatch(r"\d+(\.\d+)?", s):
            v = float(s)
            return max(v, 1.01)
    except Exception as e:
        st.warning(f"Could not parse odds string: '{s}'. Error: {e}")
        return None
    return None


def build_component_breakdown(
    primary_df, name_to_post, name_to_odds, pp_text="", name_to_prog=None
):
    """
    GOLD STANDARD: Build detailed component breakdown with complete mathematical transparency.

    Shows exactly what the rating system sees in each horse, with:
    - All component values with proper weighting (using race_class_parser if available)
    - Calculated weighted contributions
    - Full traceability of rating calculation
    - Robust error handling for missing/invalid data

    Args:
        pp_text: Full BRISNET PP text for race_class_parser analysis

    Returns: Markdown formatted component breakdown for top 5 horses
    """
    # VALIDATION: Input checks
    if primary_df is None or primary_df.empty:
        return "No component data available."

    if "Horse" not in primary_df.columns or "R" not in primary_df.columns:
        return "Invalid component data structure."

    # EXTRACTION: Get top 5 horses by rating
    try:
        top_horses = primary_df.nlargest(5, "R")
    except Exception:
        # Fallback if rating column has issues
        top_horses = primary_df.head(5)

    if top_horses.empty:
        return "No horses to analyze."

    # COMPONENT WEIGHTS: Display weights matching actual rating_engine.py computation
    # NOTE: These MUST stay in sync with compute_bias_ratings() in rating_engine.py
    # Class weight is overridden by race_class_parser when available (1.0-10.0 scale)
    WEIGHTS = {
        "Cclass": 2.5,  # Default class weight (parser overrides to 1.0-10.0)
        "Cspeed": 2.0,  # Speed figures - matches speed_multiplier default range
        "Cform": 1.8,  # Form cycle - current condition
        "Cpace": 1.5,  # Pace advantage - tactical fit
        "Cstyle": 1.2,  # Running style - bias fit
        "Cpost": 0.8,  # Post position - track bias
    }

    # IMPORTANT: Try to get actual class weight used in calculation from parser
    # This ensures breakdown shows true weights, not defaults
    if RACE_CLASS_PARSER_AVAILABLE and pp_text:
        try:
            race_class_data = parse_and_calculate_class(pp_text)
            actual_class_weight = race_class_data["weight"]["class_weight"]
            WEIGHTS["Cclass"] = actual_class_weight  # Use actual weight from parser
        except Exception:
            pass  # Fall back to default

    breakdown = "### Component Breakdown (Top 5 Horses)\n"
    # CRITICAL FIX: Use program number for horse labels (what bettors see on tote board)
    _prog_map = name_to_prog if name_to_prog else name_to_post

    breakdown += "_Mathematical transparency: Shows exactly what the system sees in each horse_\n\n"

    for _idx, row in top_horses.iterrows():
        horse_name = row.get("Horse", "Unknown")
        prog = _prog_map.get(horse_name, "?")
        ml = name_to_odds.get(horse_name, "?")

        # SAFE EXTRACTION: Get rating with error handling
        try:
            final_rating = float(row.get("R", 0))
        except Exception:
            final_rating = 0.0

        breakdown += (
            f"**#{prog} {horse_name}** (ML {ml}) - **Rating: {final_rating:.2f}**\n"
        )

        # CORE COMPONENTS: Extract with validation
        components = {}
        component_descriptions = {
            "Cclass": "Purse earnings, race level history",
            "Cform": "Recent performance trend, consistency",
            "Cspeed": "Speed figures relative to field average",
            "Cpace": "Pace advantage/disadvantage vs projected pace",
            "Cstyle": "Running style fit for pace scenario",
            "Cpost": "Post position bias for this track/distance",
        }

        weighted_sum = 0.0
        for comp_name, weight in WEIGHTS.items():
            try:
                comp_value = float(row.get(comp_name, 0))
            except Exception:
                comp_value = 0.0

            components[comp_name] = comp_value
            weighted_contribution = comp_value * weight
            weighted_sum += weighted_contribution

            description = component_descriptions.get(comp_name, "")
            breakdown += f"- **{comp_name[1:]}:** {comp_value:+.2f} (×{weight} weight = {weighted_contribution:+.2f}) - {description}\n"

        # TRACK BIAS: Additional component
        try:
            atrack = float(row.get("Atrack", 0))
        except Exception:
            atrack = 0.0
        breakdown += f"- **Track Bias:** {atrack:+.2f} - Track-specific advantages (style + post combo)\n"

        # TRANSPARENCY: Show weighted total
        breakdown += f"- **Weighted Core Total:** {weighted_sum:.2f}\n"

        # QUIRIN POINTS: BRIS pace rating
        quirin = row.get("Quirin", "N/A")
        if quirin != "N/A":
            try:
                quirin = int(float(quirin))
            except Exception:
                quirin = "N/A"
        breakdown += f"- **Quirin Points:** {quirin} - BRISNET early pace points\n"

        # FINAL RATING: Includes all angles and bonuses
        breakdown += f"- **Final Rating:** {final_rating:.2f} (includes 8 elite angles + tier 2 bonuses + track bias)\n\n"

    breakdown += "_Note: Positive values = advantages, negative = disadvantages. Weighted contributions show impact on final rating._\n"

    return breakdown


def build_betting_strategy(
    primary_df: pd.DataFrame,
    df_ol: pd.DataFrame,
    strategy_profile: str,
    name_to_post: dict[str, str],
    name_to_odds: dict[str, str],
    field_size: int,
    ppi_val: float,
    smart_money_horses: list[dict] | None = None,
    name_to_ml: dict[str, str] | None = None,
    name_to_prog: dict[str, str] | None = None,
) -> str:
    """
    Builds elite strategy report with finishing order predictions, component transparency,
    A/B/C/D grouping, and $50 bankroll optimization.

    Args:
        smart_money_horses: List of horses with significant ML→Live odds drops for Smart Money Alert
        name_to_ml: Dictionary mapping horse names to ML odds (for comparison with live odds)
        name_to_prog: Dictionary mapping horse names to program numbers (saddle cloth).
                      If None, falls back to name_to_post for backward compatibility.
    """

    # Handle None defaults for mutable default arguments
    if smart_money_horses is None:
        smart_money_horses = []
    if name_to_ml is None:
        name_to_ml = {}
    # CRITICAL FIX: Use program numbers for all horse labels/betting references
    # Program # = saddle cloth / tote board number (what bettors actually bet on)
    # Falls back to post position for backward compatibility
    if name_to_prog is None:
        name_to_prog = name_to_post

    # --- GOLD STANDARD: Build probability dictionary with validation ---
    # (Populated early so underlay/overlay detection works correctly)
    primary_probs_dict = {}
    for horse in primary_df["Horse"].tolist():
        horse_df = primary_df[primary_df["Horse"] == horse]
        if horse_df.empty:
            primary_probs_dict[horse] = 1.0 / max(len(primary_df), 1)
            continue

        prob_str = horse_df["Fair %"].iloc[0]
        try:
            # Handle multiple formats: "25.5%", "0.255", 25.5
            if isinstance(prob_str, str):
                prob = float(prob_str.strip("%").strip()) / (
                    100.0 if "%" in str(prob_str) else 1.0
                )
            else:
                prob = float(prob_str)
                if prob > 1.0:  # Assume percentage format
                    prob = prob / 100.0

            # VALIDATION: Probability bounds [0, 1]
            prob = max(0.0, min(1.0, prob))
        except Exception:
            prob = 1.0 / max(len(primary_df), 1)

        primary_probs_dict[horse] = prob

    # NORMALIZATION: Ensure probabilities sum to exactly 1.0
    total_prob = sum(primary_probs_dict.values())
    if total_prob > 0:
        primary_probs_dict = {h: p / total_prob for h, p in primary_probs_dict.items()}

    # --- ELITE: Calculate Most Likely Finishing Order (Sequential Selection Algorithm) ---
    def calculate_most_likely_finishing_order(
        df: pd.DataFrame, top_n: int = 5
    ) -> list[tuple[str, float]]:
        """
        GOLD STANDARD: Calculate most likely finishing order using mathematically sound sequential selection.

        Algorithm Guarantees:
        1. Each horse appears EXACTLY ONCE in finishing order (no duplicates)
        2. Probabilities are properly renormalized after each selection
        3. Later positions reflect conditional probabilities given earlier selections
        4. Handles edge cases (empty df, invalid probabilities, small fields)

        Mathematical Approach:
        - Position 1: Horse with highest base probability wins
        - Position 2: Remove winner, renormalize remaining field, select highest
        - Position 3-5: Continue sequential removal and selection

        Returns: List of (horse_name, conditional_probability) for positions 1-N
        """
        # VALIDATION: Input checks
        if df is None or df.empty:
            return []

        if "Horse" not in df.columns or "Fair %" not in df.columns:
            return []

        horses = df["Horse"].tolist()
        if len(horses) == 0:
            return []

        # EXTRACT: Base win probabilities with robust error handling
        win_probs = []
        for horse in horses:
            horse_df = df[df["Horse"] == horse]
            if horse_df.empty:
                win_probs.append(1.0 / len(horses))  # Fallback to uniform
                continue

            prob_str = horse_df["Fair %"].iloc[0]
            try:
                # Handle various formats: "25.5%", "0.255", 25.5
                if isinstance(prob_str, str):
                    prob = float(prob_str.strip("%").strip()) / (
                        100.0 if "%" in str(prob_str) else 1.0
                    )
                else:
                    prob = float(prob_str)
                    if prob > 1.0:  # Assume percentage
                        prob = prob / 100.0
            except Exception:
                prob = 1.0 / len(horses)

            # SANITY CHECK: Probability bounds
            prob = max(0.0, min(1.0, prob))
            win_probs.append(prob)

        win_probs = np.array(win_probs, dtype=np.float64)

        # NORMALIZATION: Ensure probabilities sum to 1.0
        if win_probs.sum() > 0:
            win_probs = win_probs / win_probs.sum()
        else:
            # All probabilities were invalid - use uniform distribution
            win_probs = np.ones(len(horses), dtype=np.float64) / len(horses)

        # CONFIDENCE CAP (TUP R6 Feb 2026): Prevent over-confident predictions
        # Failed pick #2 Ez Cowboy had 94.8% probability but finished 4th
        # Cap maximum win probability at 65% for competitive fields (6+ horses)
        if len(horses) >= 6:
            max_prob = 0.65
            for i in range(len(win_probs)):
                if win_probs[i] > max_prob:
                    excess = win_probs[i] - max_prob
                    win_probs[i] = max_prob
                    # Redistribute excess to other horses proportionally
                    other_indices = [j for j in range(len(win_probs)) if j != i]
                    if other_indices:
                        redistribution = excess / len(other_indices)
                        for j in other_indices:
                            win_probs[j] += redistribution
            # Re-normalize after capping
            if win_probs.sum() > 0:
                win_probs = win_probs / win_probs.sum()

        # SEQUENTIAL SELECTION: Build finishing order one position at a time
        finishing_order = []
        remaining_indices = list(range(len(horses)))
        remaining_probs = win_probs.copy()

        for _position in range(min(top_n, len(horses))):
            # VALIDATION: Check we have horses remaining
            if len(remaining_indices) == 0:
                break

            # RENORMALIZATION: Ensure remaining probabilities sum to 1.0
            prob_sum = remaining_probs.sum()
            if prob_sum > 0:
                remaining_probs = remaining_probs / prob_sum
            else:
                # Fallback to uniform for remaining horses
                remaining_probs = np.ones(
                    len(remaining_indices), dtype=np.float64
                ) / len(remaining_indices)

            # SELECTION: Horse with highest conditional probability
            best_relative_idx = np.argmax(remaining_probs)

            # BOUNDS CHECK: Validate index before accessing (prevents crash)
            if best_relative_idx >= len(remaining_indices):
                # Should never happen, but safety first
                break

            selected_horse_idx = remaining_indices[best_relative_idx]
            selected_prob = remaining_probs[best_relative_idx]

            # RECORD: Add to finishing order
            finishing_order.append((horses[selected_horse_idx], float(selected_prob)))

            # REMOVAL: Eliminate selected horse from remaining pool
            remaining_indices.pop(best_relative_idx)
            remaining_probs = np.delete(remaining_probs, best_relative_idx)

        return finishing_order

    # EXECUTE: Calculate most likely finishing order (ensures mathematical validity)
    finishing_order = calculate_most_likely_finishing_order(primary_df, top_n=5)

    # BUILD: Alternative horses for each position (for display purposes)
    # Shows top 3 most likely horses for each position, excluding already-selected horses
    most_likely = {}
    selected_horses = {horse for horse, _ in finishing_order}

    for pos_idx, (primary_horse, _primary_prob) in enumerate(finishing_order, start=1):
        # ALTERNATIVE CANDIDATES: Get probabilities for all horses at this position
        # Exclude horses already selected for earlier positions
        alternatives = []
        for h in primary_df["Horse"].tolist():
            # Skip horses already selected for earlier positions (except the primary for this position)
            if h in selected_horses and h != primary_horse:
                continue

            h_prob_str = primary_df[primary_df["Horse"] == h]["Fair %"].iloc[0]
            try:
                if isinstance(h_prob_str, str):
                    h_prob = float(h_prob_str.strip("%").strip()) / (
                        100.0 if "%" in str(h_prob_str) else 1.0
                    )
                else:
                    h_prob = float(h_prob_str)
                    if h_prob > 1.0:
                        h_prob = h_prob / 100.0
                h_prob = max(0.0, min(1.0, h_prob))
            except Exception:
                h_prob = 0.0

            alternatives.append((h, h_prob))

        # RANKING: Sort by probability (highest first) and take top 3
        alternatives.sort(key=lambda x: x[1], reverse=True)
        most_likely[pos_idx] = alternatives[:3]

    # --- 1. Helper Functions ---
    def format_horse_list(horse_names: list[str]) -> str:
        """Creates a bulleted list of horses with program#, name, and odds (shows ML → Live if different)."""
        if not horse_names:
            return "* None"
        lines = []
        # Sort horses by program number before displaying
        sorted_horses = sorted(
            horse_names, key=lambda name: int(name_to_prog.get(name, "999"))
        )
        for name in sorted_horses:
            prog = name_to_prog.get(name, "??")
            current_odds = name_to_odds.get(name, "N/A")
            ml_odds = name_to_ml.get(name, "")

            # Show ML → Live format when they differ, otherwise just show current odds
            if ml_odds and current_odds != ml_odds and current_odds != "N/A":
                odds_display = f"ML {ml_odds} → Live {current_odds}"
            else:
                odds_display = current_odds

            lines.append(f"* **#{prog} - {name}** ({odds_display})")
        return "\n".join(lines)

    def get_bet_cost(base: float, num_combos: int) -> str:
        """Calculates and formats simple bet cost."""
        cost = base * num_combos
        base_str = f"${base:.2f}"
        return f"{num_combos} combos = **${cost:.2f}** (at {base_str} base)"

    def get_box_combos(num_horses: int, box_size: int) -> int:
        """Calculates combinations for a box bet."""
        import math

        if num_horses < box_size:
            return 0
        return math.factorial(num_horses) // math.factorial(num_horses - box_size)

    # === FIX 1: Defined the missing helper function ===
    def get_min_cost_str(base: float, *legs) -> str:
        """Calculates part-wheel combos and cost from leg sizes."""
        final_combos = 1
        leg_counts = [l for l in legs if l > 0]  # Filter out 0s
        if not leg_counts:
            final_combos = 0
        else:
            for l in leg_counts:
                final_combos *= l

        cost = base * final_combos
        base_str = f"${base:.2f}"
        return f"{final_combos} combos = **${cost:.2f}** (at {base_str} base)"

    # ===================================================

    # --- 2. A/B/C/D Grouping Logic (with Odds Drift Gate) ---
    A_group, B_group, C_group, D_group = [], [], [], []

    # ═══════════════════════════════════════════════════════
    # CRITICAL FIX: Sort horses by RATING before grouping
    # Previously used unsorted DataFrame order (= post order)
    # which made post #1 the A-Group key regardless of rating
    # ═══════════════════════════════════════════════════════
    _sorted_for_groups = primary_df.copy()
    _sorted_for_groups["_R_numeric"] = pd.to_numeric(
        _sorted_for_groups["R"], errors="coerce"
    )
    _sorted_for_groups = _sorted_for_groups.sort_values(
        "_R_numeric", ascending=False, na_position="last"
    )
    all_horses = _sorted_for_groups["Horse"].tolist()
    pos_ev_horses = (
        set(df_ol[df_ol["EV per $1"] > 0.05]["Horse"].tolist())
        if not df_ol.empty
        else set()
    )

    # ═══════════════════════════════════════════════════════════════
    # PEGASUS 2026 TUNING: Block horses with massive odds drift from A-Group
    # British Isles 20/1 → 50/1 should NEVER have been in A-Group
    # ═══════════════════════════════════════════════════════════════
    blocked_from_A = set()
    dumb_money_horses = st.session_state.get("dumb_money_horses", [])
    if dumb_money_horses:
        for h in dumb_money_horses:
            if h.get("ratio", 1.0) > 2.0:  # 2x+ drift = blocked from A-Group
                blocked_from_A.add(h["name"])

    # Filter out blocked horses AND NaN-rated horses from A-group consideration
    # CRITICAL FIX: Never promote a NaN-rated horse to A-Group
    def _has_valid_rating(horse_name: str) -> bool:
        """Check if a horse has a valid (non-NaN, non-None) rating."""
        rows = primary_df[primary_df["Horse"] == horse_name]
        if rows.empty:
            return False
        r_val = rows["R"].iloc[0]
        if r_val is None:
            return False
        try:
            return np.isfinite(float(r_val))
        except (ValueError, TypeError):
            return False

    eligible_for_A = [
        h for h in all_horses if h not in blocked_from_A and _has_valid_rating(h)
    ]

    if strategy_profile == "Confident":
        A_group = (
            [eligible_for_A[0]]
            if eligible_for_A
            else [all_horses[0]]
            if all_horses
            else []
        )
        if len(eligible_for_A) > 1:
            second_horse = eligible_for_A[1]
            first_rating = (
                primary_df[primary_df["Horse"] == eligible_for_A[0]]["R"].iloc[0]
                if eligible_for_A
                else 0
            )
            second_rating = primary_df[primary_df["Horse"] == second_horse]["R"].iloc[0]
            try:
                first_r = float(first_rating) if first_rating is not None else 0
                second_r = float(second_rating) if second_rating is not None else 0
                if second_r > (first_r * 0.90):  # Only add #2 if very close
                    A_group.append(second_horse)
            except (ValueError, TypeError):
                pass  # Skip adding second horse if ratings are invalid
    else:  # Value Hunter - Prioritize top pick + overlays (excluding blocked horses)
        eligible_overlays = pos_ev_horses - blocked_from_A
        A_group = list(
            set([eligible_for_A[0]] if eligible_for_A else []) | eligible_overlays
        )
        if len(A_group) > 4:  # Cap A group size for Value Hunter
            A_group = sorted(
                A_group, key=lambda h: primary_df[primary_df["Horse"] == h].index[0]
            )[:4]

    B_group = [h for h in all_horses if h not in A_group][:3]
    C_group = [h for h in all_horses if h not in A_group and h not in B_group][:4]
    D_group = [
        h
        for h in all_horses
        if h not in A_group and h not in B_group and h not in C_group
    ]

    nA, nB, nC, nD = len(A_group), len(B_group), len(C_group), len(D_group)
    nAll = field_size  # Total runners

    # --- 3. Build Pace Projection ---
    pace_report = "### Pace Projection\n"
    if ppi_val > 0.5:
        pace_report += f"* **Fast Pace Likely (PPI {ppi_val:+.2f}):** Favors horses that can press or stalk ('E/P', 'P') and potentially closers ('S') if leaders tire. Pure speed ('E') might fade.\n"
    elif ppi_val < -0.5:
        pace_report += f"* **Slow Pace Likely (PPI {ppi_val:+.2f}):** Favors horses near the lead ('E', 'E/P'). Closers ('P', 'S') may find it hard to catch up.\n"
    else:
        pace_report += f"* **Moderate Pace Expected (PPI {ppi_val:+.2f}):** Fair for most styles. Tactical speed ('E/P') is often useful.\n"

    # --- 4. Build Contender Analysis (Simplified) ---
    contender_report = "### Contender Analysis\n"
    contender_report += (
        "**Key Win Contenders (A-Group):**\n" + format_horse_list(A_group) + "\n"
    )
    contender_report += "* _Primary win threats based on model rank and/or betting value. Use these horses ON TOP in exactas, trifectas, etc._\n\n"

    contender_report += (
        "**Primary Challengers (B-Group):**\n" + format_horse_list(B_group) + "\n"
    )
    contender_report += "* _Logical contenders expected to finish 2nd or 3rd. Use directly underneath A-Group horses._\n"

    top_rated_horse = all_horses[0]
    is_overlay = top_rated_horse in pos_ev_horses
    top_ml_str = name_to_odds.get(top_rated_horse, "100")
    top_ml_dec = str_to_decimal_odds(top_ml_str) or 101
    is_underlay = (
        not is_overlay
        and (primary_probs_dict.get(top_rated_horse, 0) > (1 / top_ml_dec))
        and top_ml_dec < 4
    )  # Define underlay as < 3/1

    if is_overlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_prog.get(top_rated_horse)} - {top_rated_horse}** looks like a good value bet (Overlay).\n"
    elif is_underlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_prog.get(top_rated_horse)} - {top_rated_horse}** might be overbet (Underlay at {top_ml_str}). Consider using more underneath than on top.\n"

    # --- 5. Build Simplified Blueprint Section ---
    blueprint_report = "### Betting Strategy Blueprints (Scale Base Bets to Budget: Max ~$50 Recommended)\n"
    blueprint_report += "_Costs are examples using minimum base bets ($1.00 Exacta, $0.50 Tri, $0.10 Super, $1.00 SH5). Adjust base amount to fit your total budget for this race._\n"

    # --- Field Size Logic ---
    if field_size <= 6:
        blueprint_report += "\n**Note:** With a small field (<=6 runners), Superfecta and Super High 5 payouts are often very low. Focus on Win, Exacta, and Trifecta bets.\n"
        # Generate only Win/Ex/Tri for small fields
        blueprint_report += f"\n#### {strategy_profile} Profile Plan (Small Field)\n"
        if strategy_profile == "Value Hunter":
            blueprint_report += (
                "* **Win Bets:** Consider betting all **A-Group** horses.\n"
            )
        else:  # Confident
            blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"

        # Exacta Examples
        blueprint_report += f"* **Exacta Part-Wheel:** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 2:
            ex_box_combos = get_box_combos(nA, 2)
            blueprint_report += f"* **Exacta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(1.00, ex_box_combos)}\n"

        # Trifecta Examples
        blueprint_report += f"* **Trifecta Part-Wheel:** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        if nA >= 3:
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"

        blueprint_report += (
            "_Structure: Use A-horses on top, spread underneath with B and C._\n"
        )

    else:  # Standard logic for fields > 6
        # --- Confident Blueprint ---
        blueprint_report += "\n#### Confident Profile Plan\n"
        blueprint_report += "_Focus: Key A-Group horses ON TOP._\n"
        blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B` ({nA}x{nB}) - {get_min_cost_str(1.00, nA, nB)}\n"
        if (
            nA >= 1 and nB >= 1 and nC >= 1
        ):  # Check if groups have members for straight example
            blueprint_report += "* **Straight Trifecta:** `Top A / Top B / Top C` (1 combo) - Consider at higher base (e.g., $1 or $2).\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B / C / D` ({nA}x{nB}x{nC}x{nD}) - {get_min_cost_str(0.10, nA, nB, nC, nD)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B / C / D / ALL` ({nA}x{nB}x{nC}x{nD}x{nAll}) - {get_min_cost_str(1.00, nA, nB, nC, nD, nAll)}\n"

        # --- Value-Hunter Blueprint ---
        blueprint_report += "\n#### Value-Hunter Profile Plan\n"
        blueprint_report += "_Focus: Use A-Group (includes overlays) ON TOP, spread wider underneath._\n"
        blueprint_report += "* **Win Bets:** Consider betting all **A-Group** horses.\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 3:  # Example box if A group is large enough
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B,C / B,C,D` ({nA}x{nB + nC}x{nB + nC + nD}) - {get_min_cost_str(0.50, nA, nB + nC, nB + nC + nD)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B,C / B,C,D / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B,C / B,C,D / ALL / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}x{nAll}) - {get_min_cost_str(1.00, nA, nB + nC, nB + nC + nD, nAll, nAll)}\n"

    detailed_breakdown = build_component_breakdown(
        primary_df,
        name_to_post,
        name_to_odds,
        pp_text=pp_text,
        name_to_prog=name_to_prog,
    )

    component_report = "### What Our System Sees in Top Contenders\n\n"
    component_report += detailed_breakdown + "\n"

    # --- GOLD STANDARD: Build Finishing Order Predictions (NO DUPLICATES) ---
    # Use sequential selection algorithm - each horse appears EXACTLY ONCE
    finishing_order = calculate_most_likely_finishing_order(primary_df, top_n=5)

    finishing_order_report = "### Most Likely Finishing Order\n\n"
    finishing_order_report += "**Algorithm:** Sequential selection ensuring each horse appears EXACTLY ONCE. The percentage indicates conditional probability (e.g., for 2nd place: probability of finishing 2nd given not finishing 1st).\n\n"

    position_names = {
        1: "🥇 Win (1st)",
        2: "🥈 Place (2nd)",
        3: "🥉 Show (3rd)",
        4: "4th",
        5: "5th",
    }

    for pos, (horse, prob) in enumerate(finishing_order, 1):
        prog = name_to_prog.get(horse, "?")
        odds = name_to_odds.get(horse, "?")
        finishing_order_report += f"* **{position_names[pos]} • #{prog} {horse}** (Odds: {odds}) — {prob * 100:.1f}% conditional probability\n"

    finishing_order_report += "\n💡 **Use These Rankings:** Build your exotic tickets using this exact finishing order for optimal probability-based coverage.\n\n"

    # --- ELITE: Build $50 Bankroll Optimization ---
    bankroll_report = "### $50 Bankroll Structure\n\n"

    if strategy_profile == "Value Hunter":
        bankroll_report += (
            "**Strategy:** Value Hunter - Focus on overlays with wider coverage\n\n"
        )

        # Win bets on A-Group overlays
        win_cost = min(len([h for h in A_group if h in pos_ev_horses]), 3) * 8
        bankroll_report += f"* **Win Bets** (${win_cost}): $8 each on top {min(len([h for h in A_group if h in pos_ev_horses]), 3)} overlay(s) from A-Group\n"

        # Exacta part-wheel
        ex_combos = nA * (nB + min(nC, 2))
        ex_cost = min(int(ex_combos * 0.50), 14)
        bankroll_report += f"* **Exacta** (${ex_cost}): A / B,C (top 2 from C) - ${ex_cost / ex_combos:.2f} base × {ex_combos} combos\n"

        # Trifecta
        tri_combos = nA * (nB + min(nC, 2)) * (nB + nC + min(nD, 2))
        tri_cost = min(int(tri_combos * 0.30), 12)
        bankroll_report += f"* **Trifecta** (${tri_cost}): A / B,C / B,C,D - ${tri_cost / tri_combos:.2f} base × {tri_combos} combos\n"

        # Superfecta
        super_cost = 50 - win_cost - ex_cost - tri_cost
        super_cost = max(super_cost, 8)
        bankroll_report += (
            f"* **Superfecta** (${super_cost}): A / B,C / B,C,D / Top 5 from D+All\n"
        )

    else:  # Confident
        bankroll_report += (
            "**Strategy:** Confident - Focus on top pick with deeper coverage\n\n"
        )

        # Win bet on top pick
        bankroll_report += f"* **Win Bet** ($20): $20 on #{name_to_prog.get(all_horses[0], '?')} {all_horses[0]}\n"

        # Exacta
        ex_combos = nA * nB
        bankroll_report += f"* **Exacta** ($10): A / B - ${10 / max(ex_combos, 1):.2f} base × {ex_combos} combos\n"

        # Trifecta
        tri_combos = nA * nB * nC
        bankroll_report += f"* **Trifecta** ($12): A / B / C - ${12 / max(tri_combos, 1):.2f} base × {tri_combos} combos\n"

        # Superfecta
        bankroll_report += (
            "* **Superfecta** ($8): A / B / C / D - scaled to fit budget\n"
        )

    bankroll_report += "\n**Total Investment:** $50 (optimized)\n"
    bankroll_report += f"**Risk Level:** {strategy_profile} approach - {'Wider coverage, value-based' if strategy_profile == 'Value Hunter' else 'Concentrated on top selection'}\n"
    bankroll_report += "\n💡 **Use Finishing Order Predictions:** The probability rankings above show the most likely finishers for each position. Build your tickets using horses with highest probabilities for each slot.\n"

    # --- Smart Money Alert Section ---
    smart_money_report = ""
    if smart_money_horses:
        smart_money_report = (
            "### 🚨 Smart Money Alert: Significant Odds Movement Detected\n\n"
        )
        smart_money_report += "The following horses show **major public support** with Live odds dropping significantly from Morning Line. This typically indicates:\n"
        smart_money_report += "* Trainer/connections betting\n"
        smart_money_report += "* Informed money from sharp handicappers\n"
        smart_money_report += "* Positive workout reports or stable buzz\n"
        smart_money_report += "* Hidden form improvement\n\n"

        # Sort by biggest movement percentage
        smart_money_horses.sort(key=lambda x: x["movement_pct"], reverse=True)

        for horse_data in smart_money_horses:
            name = horse_data["name"]
            post = horse_data["post"]
            ml = horse_data["ml"]
            live = horse_data["live"]
            movement_pct = horse_data["movement_pct"]

            smart_money_report += f"* **🔥 #{post} {name}** - ML {ml} → Live {live} (📉 {movement_pct:.0f}% drop)\n"

        smart_money_report += "\n💡 **Action:** These horses are getting **heavy public support**. Consider them seriously even if model doesn't rank them #1. Sharp money often spots angles the numbers miss.\n\n"
        smart_money_report += "---\n"

    # --- 6. Build Final Report String (OPTIMIZED ORDER: Most Important First) ---
    final_report = f"""
{finishing_order_report}
---
{smart_money_report}{bankroll_report}
---
{component_report}
---
{contender_report}
---
### A/B/C/D Contender Groups for Tickets

**A-Group (Key Win Contenders - Use ON TOP)**
{format_horse_list(A_group)}

**B-Group (Primary Challengers - Use 2nd/3rd)**
{format_horse_list(B_group)}

**C-Group (Underneath Keys - Use 2nd/3rd/4th)**
{format_horse_list(C_group)}

**D-Group (Exotic Fillers - Use 3rd/4th/5th)**
{format_horse_list(D_group)}

---
{pace_report}
---
{blueprint_report}
---
### Bankroll & Strategy Notes
* **Budget:** Decide your total wager amount for this race (e.g., $20, $50 recommended max per bet type).
* **Scale Base Bets:** Exacta $1.00, Trifecta $0.50–$1.00, Superfecta $0.10–$0.20, Super High-5 $1.00. Adjust to match your budget.
* **Confidence:** Bet more confidently when A/B groups look strong. Reduce base bets or narrow the C/D groups in tickets if less confident.
* **Small Fields (<=6):** Focus on Win/Exacta/Trifecta as complex exotics pay less.
* **Play SH5** mainly on mandatory payout days or when you have a very strong opinion & budget allows.
"""
    return final_report
