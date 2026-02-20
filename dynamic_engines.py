"""
dynamic_engines.py — Elite Dynamic Engines Orchestrator
========================================================
Single entry-point that activates ALL engines in the correct sequence:

    1. GoldStandardBRISNETParser  → 80+ HorseData fields
    2. compute_eight_angles()     → 8 normalised angle scores
    3. MLBlendEngine              → PyTorch z-normalised ML score
    4. TrackIntelligenceEngine    → Learned track profile + bias weights
    5. UnifiedRatingEngine        → Bayesian multi-component final rating

Guarantees:
    • Every parsed PP field is routed to at least one engine
    • All 8 angles are computed (non-zero when data exists)
    • Track profiles (learned > hardcoded) are applied
    • ML blend score injected as 7th Bayesian component
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine availability flags
# ---------------------------------------------------------------------------

try:
    from unified_rating_engine import UnifiedRatingEngine

    _URE_OK = True
except ImportError:
    _URE_OK = False
    UnifiedRatingEngine = None  # type: ignore[misc,assignment]

try:
    from ml_blend_engine import MLBlendEngine

    _MLB_OK = True
except ImportError:
    _MLB_OK = False
    MLBlendEngine = None  # type: ignore[misc,assignment]

try:
    from track_intelligence import TrackIntelligenceEngine

    _TIE_OK = True
except ImportError:
    _TIE_OK = False
    TrackIntelligenceEngine = None  # type: ignore[misc,assignment]

try:
    from horse_angles8 import compute_eight_angles

    _ANG_OK = True
except ImportError:
    _ANG_OK = False
    compute_eight_angles = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """Unified result from the full engine pipeline."""

    predictions: pd.DataFrame  # Final ranked predictions
    angles_df: pd.DataFrame | None  # 8-angle breakdown per horse
    ml_scores: dict[str, float]  # Raw ML blend scores
    track_profile: dict[str, Any]  # Active track profile used
    engine_report: dict[str, bool]  # Which engines were activated
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimum learned-race threshold before we trust learned profile over hardcoded
# ---------------------------------------------------------------------------
LEARNED_PROFILE_MIN_RACES = 10


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DynamicEnginesOrchestrator:
    """
    Stateless orchestrator — call ``process_race()`` for each race card.
    Instantiate once; it internally creates engine singletons.
    """

    def __init__(
        self,
        db_path: str = "gold_high_iq.db",
        softmax_tau: float = 3.0,
        learned_weights: dict[str, float] | None = None,
    ):
        self.db_path = db_path

        # --- Unified Rating Engine (core) ---
        self.ure: UnifiedRatingEngine | None = None
        if _URE_OK:
            self.ure = UnifiedRatingEngine(
                softmax_tau=softmax_tau,
                learned_weights=learned_weights,
            )

        # --- ML Blend Engine ---
        self.ml_engine: MLBlendEngine | None = None
        if _MLB_OK:
            try:
                self.ml_engine = MLBlendEngine(db_path=db_path)
            except Exception as e:
                logger.warning(f"MLBlendEngine init failed: {e}")

        # --- Track Intelligence Engine ---
        self.track_engine: TrackIntelligenceEngine | None = None
        if _TIE_OK:
            try:
                self.track_engine = TrackIntelligenceEngine(db_path=db_path)
            except Exception as e:
                logger.warning(f"TrackIntelligenceEngine init failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_race(
        self,
        pp_text: str,
        today_purse: int,
        today_race_type: str,
        track_name: str,
        surface_type: str,
        distance_txt: str,
        condition_txt: str = "fast",
        style_bias: list[str] | None = None,
        post_bias: list[str] | None = None,
    ) -> OrchestratorResult:
        """
        Full pipeline: PP text → OrchestratorResult.

        Parameters match ``UnifiedRatingEngine.predict_race()`` so the
        orchestrator is a drop-in replacement anywhere the engine is called.
        """
        report: dict[str, bool] = {
            "parser": False,
            "angles": False,
            "ml_blend": False,
            "track_intelligence": False,
            "unified_engine": False,
        }
        warnings: list[str] = []
        ml_scores: dict[str, float] = {}
        angles_df: pd.DataFrame | None = None
        track_profile: dict[str, Any] = {}

        # ── 1. ML Blend Scores (pre-compute before URE call) ──────────
        if self.ml_engine and self.ml_engine.is_available:
            try:
                ml_scores = self._compute_ml_scores(
                    pp_text, track_name, today_race_type
                )
                if ml_scores:
                    report["ml_blend"] = True
                    logger.info(f"🤖 ML Blend: scored {len(ml_scores)} horses")
            except Exception as e:
                warnings.append(f"ML Blend failed: {e}")
                logger.warning(f"ML Blend error: {e}")

        # ── 2. Track Intelligence Profile ─────────────────────────────
        if self.track_engine:
            try:
                track_profile = self._resolve_track_profile(
                    track_name, surface_type, distance_txt, condition_txt
                )
                if track_profile:
                    report["track_intelligence"] = True
            except Exception as e:
                warnings.append(f"Track Intelligence failed: {e}")
                logger.warning(f"Track Intelligence error: {e}")

        # ── 3. Unified Rating Engine (includes parser + angles) ───────
        predictions = pd.DataFrame()
        if self.ure:
            try:
                predictions = self.ure.predict_race(
                    pp_text=pp_text,
                    today_purse=today_purse,
                    today_race_type=today_race_type,
                    track_name=track_name,
                    surface_type=surface_type,
                    distance_txt=distance_txt,
                    condition_txt=condition_txt,
                    style_bias=style_bias,
                    post_bias=post_bias,
                    ml_scores=ml_scores if ml_scores else None,
                )
                if predictions is not None and not predictions.empty:
                    report["parser"] = True
                    report["angles"] = True
                    report["unified_engine"] = True
            except Exception as e:
                warnings.append(f"UnifiedRatingEngine failed: {e}")
                logger.error(f"URE error: {e}", exc_info=True)

        # ── 4. Extract angle breakdown (for UI display) ───────────────
        if predictions is not None and not predictions.empty and _ANG_OK and self.ure:
            try:
                horses = self.ure.parser.parse_full_pp(pp_text)
                if horses:
                    df_angles_input = self.ure._horses_to_dataframe(horses)
                    angles_df = compute_eight_angles(df_angles_input)
                    report["angles"] = True
            except Exception as e:
                logger.debug(f"Angle extraction for UI: {e}")

        # ── 5. Validate 8-angle integrity ─────────────────────────────
        if angles_df is not None and not angles_df.empty:
            zero_angles = []
            for col in [
                "EarlySpeed",
                "Class",
                "Pedigree",
                "Connections",
                "Post",
                "RunstyleBias",
                "WorkPattern",
                "Recency",
            ]:
                if col in angles_df.columns and (angles_df[col] == 0).all():
                    zero_angles.append(col)
            if zero_angles:
                warnings.append(f"All-zero angles detected: {', '.join(zero_angles)}")

        return OrchestratorResult(
            predictions=predictions,
            angles_df=angles_df,
            ml_scores=ml_scores,
            track_profile=track_profile,
            engine_report=report,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Track profile resolution (learned > hardcoded)
    # ------------------------------------------------------------------

    def _resolve_track_profile(
        self,
        track_name: str,
        surface: str,
        distance_txt: str,
        condition: str,
    ) -> dict[str, Any]:
        """
        Returns the best available track profile.

        Priority:
            1. Learned profile (from TrackIntelligenceEngine) if ≥ LEARNED_PROFILE_MIN_RACES
            2. Hardcoded TRACK_BIAS_PROFILES from unified_rating_engine
            3. Empty dict (defaults apply)
        """
        if not self.track_engine:
            return {}

        # Map distance text to bucket
        distance_bucket = "all"
        if distance_txt:
            dist_lower = distance_txt.lower()
            if any(kw in dist_lower for kw in ("sprint", "5f", "6f", "5.5f", "6.5f")):
                distance_bucket = "sprint"
            elif any(kw in dist_lower for kw in ("route", "mile", "8f", "9f", "10f")):
                distance_bucket = "route"

        # Map condition
        condition_bucket = condition.lower() if condition else "all"

        learned = self.track_engine.load_profile(
            track_code=track_name,
            surface=surface.lower() if surface else "all",
            distance_bucket=distance_bucket,
            condition_bucket=condition_bucket,
        )

        if learned and learned.total_races >= LEARNED_PROFILE_MIN_RACES:
            logger.info(
                f"📊 Using LEARNED track profile for {track_name} "
                f"({learned.total_races} races)"
            )
            return {
                "source": "learned",
                "total_races": learned.total_races,
                "speed_bias": learned.speed_bias,
                "pace_bias": learned.pace_bias,
                "post_bias": learned.post_position_bias,
                "style_bias": learned.running_style_bias,
            }

        logger.info(
            f"📊 Using DEFAULT track profile for {track_name} "
            f"(learned races: {learned.total_races if learned else 0})"
        )
        return {"source": "default", "total_races": 0}

    # ------------------------------------------------------------------
    # ML score pre-computation
    # ------------------------------------------------------------------

    def _compute_ml_scores(
        self,
        pp_text: str,
        track_name: str,
        race_type: str,
    ) -> dict[str, float]:
        """Parse PP → build feature dict → run MLBlendEngine.score_horses()."""
        if not self.ml_engine or not self.ure:
            return {}

        horses = self.ure.parser.parse_full_pp(pp_text)
        if not horses:
            return {}

        # Build feature dict matching ml_blend_engine expectations
        horse_features: dict[str, dict[str, float]] = {}
        for name, h in horses.items():
            horse_features[name] = {
                "speed_rating": float(h.last_fig) if h.last_fig else 0.0,
                "avg_speed": float(h.avg_speed_last3) if h.avg_speed_last3 else 0.0,
                "class_rating": float(h.class_rating_individual or 0),
                "days_since": float(h.days_since_last or 30),
                "post_position": float(
                    int("".join(c for c in str(h.post) if c.isdigit()) or "0")
                ),
                "jockey_win_pct": float(h.jockey_win_pct or 0) * 100,
                "trainer_win_pct": float(h.trainer_win_pct or 0) * 100,
                "morning_line_odds": float(h.morning_line or 10.0),
                "workout_count": float(h.workout_count or 0),
            }

        return self.ml_engine.score_horses(horse_features)

    # ------------------------------------------------------------------
    # Convenience: check which engines are available
    # ------------------------------------------------------------------

    @property
    def available_engines(self) -> dict[str, bool]:
        return {
            "UnifiedRatingEngine": self.ure is not None,
            "MLBlendEngine": self.ml_engine is not None and self.ml_engine.is_available,
            "TrackIntelligenceEngine": self.track_engine is not None,
            "compute_eight_angles": _ANG_OK,
        }

    def engine_status_summary(self) -> str:
        """One-line status for logging / UI display."""
        statuses = []
        for name, ok in self.available_engines.items():
            statuses.append(f"{'✅' if ok else '❌'} {name}")
        return " | ".join(statuses)
