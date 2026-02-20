"""
TRACK INTELLIGENCE MODULE — Bias Detection & Profile Management
================================================================

Provides automatic bias detection, track-specific profiling, and
insight generation for the Gold High-IQ system.

Key capabilities:
  - Surface / distance / condition bucket accuracy stats
  - Running style bias detection (speed / presser / closer dominance)
  - Post position bias analysis (inside / middle / outside)
  - Jockey-Trainer combo strength (top 20 recurring combos per track)
  - Automatic textual insight generation for dashboard cards

All data is derived from the gold_high_iq + horses_analyzed tables
and stored in track_intelligence_profiles for fast lookup.

Author: Senior Horse Racing Analytics Architect
Date: February 20, 2026
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════


@dataclass
class BiasInsight:
    """A single detected bias insight for display."""

    category: str  # "style", "post", "surface", "jockey_trainer"
    severity: str  # "strong", "moderate", "mild"
    description: str  # Human-readable insight
    stat: float  # Key statistic (e.g., win %)
    sample_size: int  # Number of races supporting this insight
    confidence: float  # 0-1 confidence based on sample size


@dataclass
class TrackProfile:
    """Complete intelligence profile for one track."""

    track_code: str
    total_races: int = 0
    total_horses: int = 0

    # Surface accuracy
    dirt_races: int = 0
    dirt_winner_pct: float = 0.0
    dirt_top3_pct: float = 0.0
    turf_races: int = 0
    turf_winner_pct: float = 0.0
    turf_top3_pct: float = 0.0
    synthetic_races: int = 0

    # Distance accuracy
    sprint_races: int = 0  # < 8f
    sprint_winner_pct: float = 0.0
    route_races: int = 0  # >= 8f
    route_winner_pct: float = 0.0

    # Condition accuracy
    fast_races: int = 0
    fast_winner_pct: float = 0.0
    off_track_races: int = 0
    off_track_winner_pct: float = 0.0

    # Style bias
    style_bias: str = "Neutral"
    speed_win_pct: float = 0.0
    presser_win_pct: float = 0.0
    closer_win_pct: float = 0.0

    # Post bias
    inside_win_pct: float = 0.0  # Posts 1-3
    inside_top4_pct: float = 0.0
    middle_win_pct: float = 0.0  # Posts 4-6
    middle_top4_pct: float = 0.0
    outside_win_pct: float = 0.0  # Posts 7+
    outside_top4_pct: float = 0.0

    # Top jockey-trainer combos
    top_jt_combos: list = field(default_factory=list)

    # Overall accuracy
    overall_winner_pct: float = 0.0
    overall_top3_pct: float = 0.0
    overall_top4_pct: float = 0.0

    # Detected biases
    insights: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# DISTANCE CLASSIFICATION
# ═══════════════════════════════════════════════════════════


def classify_distance(distance) -> str:
    """Classify distance into sprint/route/marathon bucket."""
    try:
        d = float(distance)
    except (ValueError, TypeError):
        return "unknown"

    if d < 8.0:
        return "sprint"
    elif d <= 12.0:
        return "route"
    else:
        return "marathon"


def classify_condition(condition: str) -> str:
    """Classify track condition into fast/off categories."""
    if not condition:
        return "unknown"
    cond = condition.strip().lower()
    if cond in ("fast", "firm", "good"):
        return "fast"
    elif cond in ("sloppy", "muddy", "yielding", "soft", "heavy", "wet fast"):
        return "off_track"
    else:
        return "unknown"


def classify_post(post: int, field_size: int) -> str:
    """Classify post position into inside/middle/outside."""
    if field_size <= 0:
        return "middle"
    ratio = post / field_size
    if ratio <= 0.33:
        return "inside"
    elif ratio <= 0.66:
        return "middle"
    else:
        return "outside"


def classify_style(style: str) -> str:
    """Normalise running style to speed/presser/closer."""
    if not style:
        return "unknown"
    s = style.strip().upper()
    if s in ("E", "EP", "E/P", "E/EP"):
        return "speed"
    elif s in ("P", "PR", "PRESS", "PRESSER"):
        return "presser"
    elif s in ("S", "C", "CS", "CLOSER", "DEEP CLOSER"):
        return "closer"
    else:
        return "unknown"


# ═══════════════════════════════════════════════════════════
# SCHEMA: track_intelligence_profiles
# ═══════════════════════════════════════════════════════════

TRACK_INTELLIGENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_intelligence_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_code TEXT NOT NULL,
    surface TEXT NOT NULL DEFAULT 'all',
    distance_bucket TEXT NOT NULL DEFAULT 'all',
    condition_bucket TEXT NOT NULL DEFAULT 'all',
    profile_json TEXT NOT NULL,
    insights_json TEXT,
    total_races INTEGER DEFAULT 0,
    last_updated TEXT,
    UNIQUE(track_code, surface, distance_bucket, condition_bucket)
)
"""

JT_COMBOS_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_jt_combos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_code TEXT NOT NULL,
    jockey TEXT NOT NULL,
    trainer TEXT NOT NULL,
    starts INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    top3 INTEGER DEFAULT 0,
    win_pct REAL DEFAULT 0.0,
    roi REAL DEFAULT 0.0,
    last_updated TEXT,
    UNIQUE(track_code, jockey, trainer)
)
"""

TRACK_ML_PROFILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_ml_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_code TEXT NOT NULL,
    model_path TEXT,
    n_features INTEGER,
    hidden_dim INTEGER DEFAULT 64,
    val_winner_acc REAL DEFAULT 0.0,
    val_top3_acc REAL DEFAULT 0.0,
    val_top4_acc REAL DEFAULT 0.0,
    races_trained INTEGER DEFAULT 0,
    last_retrained TEXT,
    blend_weight REAL DEFAULT 1.5,
    UNIQUE(track_code)
)
"""


# ═══════════════════════════════════════════════════════════
# TRACK INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════


class TrackIntelligenceEngine:
    """
    Analyses gold_high_iq data per-track and generates rich bias
    profiles, accuracy stats, and textual insights.

    Updates incrementally after each race result submission.
    """

    # Minimum sample sizes for confidence levels
    MIN_CONFIDENT = 20  # "confident" bias detection
    MIN_MODERATE = 10  # "moderate" detection
    MIN_MILD = 5  # "mild" detection

    def __init__(self, db_path: str = "gold_high_iq.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """Create intelligence tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(TRACK_INTELLIGENCE_SCHEMA)
            cursor.execute(JT_COMBOS_SCHEMA)
            cursor.execute(TRACK_ML_PROFILES_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Track intelligence schema init failed: {e}")

    def build_full_profile(self, track_code: str) -> TrackProfile:
        """
        Build a complete intelligence profile for one track from DB data.

        Queries gold_high_iq for all races at this track, computes:
        - Surface/distance/condition accuracy breakdowns
        - Running style bias
        - Post position bias
        - Top jockey-trainer combos
        - Auto-generated bias insights

        Args:
            track_code: Track name (e.g., "Oaklawn Park")

        Returns:
            TrackProfile with all fields populated
        """
        profile = TrackProfile(track_code=track_code)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ── Load all race results for this track ─────────────
        cursor.execute(
            """
            SELECT g.race_id, g.horse_name, g.actual_finish_position,
                   g.predicted_finish_position, g.post_position, g.field_size,
                   g.surface, g.distance, g.running_style, g.odds,
                   r.track_condition, g.race_type,
                   COALESCE(h.jockey, g.jockey) AS jockey,
                   COALESCE(h.trainer, g.trainer) AS trainer
            FROM gold_high_iq g
            LEFT JOIN races_analyzed r ON g.race_id = r.race_id
            LEFT JOIN horses_analyzed h ON g.race_id = h.race_id AND g.horse_name = h.horse_name
            WHERE UPPER(g.track) = UPPER(?)
            ORDER BY g.race_id, g.actual_finish_position
        """,
            (track_code,),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return profile

        # ── Organise into races ──────────────────────────────
        races = defaultdict(list)
        for row in rows:
            race_id = row[0]
            races[race_id].append(
                {
                    "horse_name": row[1],
                    "actual": row[2],
                    "predicted": row[3],
                    "post": row[4] or 0,
                    "field_size": row[5] or 0,
                    "surface": (row[6] or "Dirt").strip().title(),
                    "distance": row[7],
                    "style": row[8] or "",
                    "odds": row[9] or 0,
                    "condition": row[10] or "",
                    "race_type": row[11] or "",
                    "jockey": row[12] or "",
                    "trainer": row[13] or "",
                }
            )

        profile.total_races = len(races)
        profile.total_horses = len(rows)

        # ── Surface / Distance / Condition accuracy ──────────
        surface_stats = defaultdict(
            lambda: {"winner_correct": 0, "top3_correct": 0, "total": 0}
        )
        distance_stats = defaultdict(
            lambda: {"winner_correct": 0, "top3_correct": 0, "total": 0}
        )
        condition_stats = defaultdict(
            lambda: {"winner_correct": 0, "top3_correct": 0, "total": 0}
        )
        style_stats = defaultdict(lambda: {"wins": 0, "total_starts": 0})
        post_stats = defaultdict(lambda: {"wins": 0, "top4": 0, "total": 0})
        jt_stats = defaultdict(lambda: {"starts": 0, "wins": 0, "top3": 0})

        overall_winner_correct = 0
        overall_top3_correct = 0
        overall_top4_correct = 0

        for race_id, horses in races.items():
            if len(horses) < 2:
                continue

            # Race-level metadata from first horse
            surface = horses[0]["surface"]
            distance = horses[0]["distance"]
            condition = horses[0]["condition"]
            dist_bucket = classify_distance(distance)
            cond_bucket = classify_condition(condition)

            # Winner prediction accuracy (was predicted #1 the actual winner?)
            actual_winner = min(
                horses,
                key=lambda h: h["actual"] if h["actual"] and h["actual"] > 0 else 999,
            )
            pred_winner = min(
                horses,
                key=lambda h: (
                    h["predicted"] if h["predicted"] and h["predicted"] > 0 else 999
                ),
            )
            winner_correct = (
                1 if actual_winner["horse_name"] == pred_winner["horse_name"] else 0
            )

            # Top-3 overlap
            actual_top3 = set(
                h["horse_name"]
                for h in sorted(horses, key=lambda h: h["actual"] or 999)[:3]
            )
            pred_top3 = set(
                h["horse_name"]
                for h in sorted(horses, key=lambda h: h["predicted"] or 999)[:3]
            )
            top3_correct = len(actual_top3 & pred_top3) / 3.0

            # Top-4 overlap
            actual_top4 = set(
                h["horse_name"]
                for h in sorted(horses, key=lambda h: h["actual"] or 999)[:4]
            )
            pred_top4 = set(
                h["horse_name"]
                for h in sorted(horses, key=lambda h: h["predicted"] or 999)[:4]
            )
            top4_correct = len(actual_top4 & pred_top4) / 4.0

            overall_winner_correct += winner_correct
            overall_top3_correct += top3_correct
            overall_top4_correct += top4_correct

            # Surface breakdown
            surface_stats[surface]["winner_correct"] += winner_correct
            surface_stats[surface]["top3_correct"] += top3_correct
            surface_stats[surface]["total"] += 1

            # Distance breakdown
            distance_stats[dist_bucket]["winner_correct"] += winner_correct
            distance_stats[dist_bucket]["top3_correct"] += top3_correct
            distance_stats[dist_bucket]["total"] += 1

            # Condition breakdown
            condition_stats[cond_bucket]["winner_correct"] += winner_correct
            condition_stats[cond_bucket]["top3_correct"] += top3_correct
            condition_stats[cond_bucket]["total"] += 1

            # Style bias — which running style wins?
            for h in horses:
                style_cat = classify_style(h["style"])
                if style_cat != "unknown":
                    style_stats[style_cat]["total_starts"] += 1
                    if h["actual"] == 1:
                        style_stats[style_cat]["wins"] += 1

            # Post position bias — which zone wins?
            for h in horses:
                post = h["post"]
                fs = h["field_size"] or len(horses)
                zone = classify_post(post, fs)
                post_stats[zone]["total"] += 1
                if h["actual"] == 1:
                    post_stats[zone]["wins"] += 1
                if h["actual"] and h["actual"] <= 4:
                    post_stats[zone]["top4"] += 1

            # Jockey-Trainer combos
            for h in horses:
                jockey = h["jockey"]
                trainer = h["trainer"]
                if jockey and trainer:
                    key = (jockey, trainer)
                    jt_stats[key]["starts"] += 1
                    if h["actual"] == 1:
                        jt_stats[key]["wins"] += 1
                    if h["actual"] and h["actual"] <= 3:
                        jt_stats[key]["top3"] += 1

        # ── Populate profile fields ──────────────────────────
        n = profile.total_races
        if n > 0:
            profile.overall_winner_pct = (overall_winner_correct / n) * 100
            profile.overall_top3_pct = (overall_top3_correct / n) * 100
            profile.overall_top4_pct = (overall_top4_correct / n) * 100

        # Surface
        for surface_name, key_name in [("Dirt", "dirt"), ("Turf", "turf")]:
            s = surface_stats.get(surface_name, {"total": 0})
            setattr(profile, f"{key_name}_races", s["total"])
            if s["total"] > 0:
                setattr(
                    profile,
                    f"{key_name}_winner_pct",
                    (s["winner_correct"] / s["total"]) * 100,
                )
                setattr(
                    profile,
                    f"{key_name}_top3_pct",
                    (s["top3_correct"] / s["total"]) * 100,
                )

        # Distance
        for bucket in ["sprint", "route"]:
            d = distance_stats.get(bucket, {"total": 0})
            setattr(profile, f"{bucket}_races", d["total"])
            if d["total"] > 0:
                setattr(
                    profile,
                    f"{bucket}_winner_pct",
                    (d["winner_correct"] / d["total"]) * 100,
                )

        # Condition
        for cond in ["fast", "off_track"]:
            c = condition_stats.get(cond, {"total": 0})
            setattr(profile, f"{cond}_races", c["total"])
            if c["total"] > 0:
                setattr(
                    profile,
                    f"{cond}_winner_pct",
                    (c["winner_correct"] / c["total"]) * 100,
                )

        # Style bias
        total_style_wins = sum(v["wins"] for v in style_stats.values())
        if total_style_wins > 0:
            for style_cat in ["speed", "presser", "closer"]:
                s = style_stats.get(style_cat, {"wins": 0, "total_starts": 0})
                win_pct = (
                    (s["wins"] / s["total_starts"] * 100)
                    if s["total_starts"] > 0
                    else 0
                )
                setattr(profile, f"{style_cat}_win_pct", win_pct)

            # Determine dominant style
            style_win_rates = {
                "speed": profile.speed_win_pct,
                "presser": profile.presser_win_pct,
                "closer": profile.closer_win_pct,
            }
            dominant = max(style_win_rates, key=style_win_rates.get)
            if (
                style_win_rates[dominant]
                > style_win_rates.get(min(style_win_rates, key=style_win_rates.get), 0)
                * 1.5
            ):
                profile.style_bias = f"Strong {dominant.title()} Bias"
            elif style_win_rates[dominant] > 0:
                profile.style_bias = f"Moderate {dominant.title()} Bias"
            else:
                profile.style_bias = "Neutral"

        # Post position bias
        for zone in ["inside", "middle", "outside"]:
            p = post_stats.get(zone, {"wins": 0, "top4": 0, "total": 0})
            if p["total"] > 0:
                setattr(profile, f"{zone}_win_pct", (p["wins"] / p["total"]) * 100)
                setattr(profile, f"{zone}_top4_pct", (p["top4"] / p["total"]) * 100)

        # Top J/T combos (min 3 starts)
        jt_ranked = sorted(
            [(k, v) for k, v in jt_stats.items() if v["starts"] >= 3],
            key=lambda x: x[1]["wins"] / max(x[1]["starts"], 1),
            reverse=True,
        )[:20]
        profile.top_jt_combos = [
            {
                "jockey": k[0],
                "trainer": k[1],
                "starts": v["starts"],
                "wins": v["wins"],
                "win_pct": round(v["wins"] / v["starts"] * 100, 1)
                if v["starts"] > 0
                else 0,
            }
            for k, v in jt_ranked
        ]

        # ── Generate bias insights ───────────────────────────
        profile.insights = self._detect_biases(profile, post_stats, style_stats)

        return profile

    def _detect_biases(
        self, profile: TrackProfile, post_stats: dict, style_stats: dict
    ) -> list[BiasInsight]:
        """Auto-detect actionable biases from the profile statistics."""
        insights = []

        # ── Style bias ───────────────────────────────────────
        total_style_starts = sum(v["total_starts"] for v in style_stats.values())
        if total_style_starts >= self.MIN_MILD:
            for style_cat in ["speed", "presser", "closer"]:
                s = style_stats.get(style_cat, {"wins": 0, "total_starts": 0})
                if s["total_starts"] >= self.MIN_MILD:
                    win_pct = s["wins"] / s["total_starts"] * 100
                    # Baseline expectation ~33% if equal
                    if win_pct >= 50:
                        severity = (
                            "strong"
                            if s["total_starts"] >= self.MIN_CONFIDENT
                            else "moderate"
                        )
                        insights.append(
                            BiasInsight(
                                category="style",
                                severity=severity,
                                description=f"{style_cat.title()} runners win {win_pct:.0f}% of races",
                                stat=win_pct,
                                sample_size=s["total_starts"],
                                confidence=min(s["total_starts"] / 30, 1.0),
                            )
                        )
                    elif win_pct <= 10 and s["total_starts"] >= self.MIN_MODERATE:
                        insights.append(
                            BiasInsight(
                                category="style",
                                severity="moderate",
                                description=f"{style_cat.title()} runners rarely win ({win_pct:.0f}%)",
                                stat=win_pct,
                                sample_size=s["total_starts"],
                                confidence=min(s["total_starts"] / 30, 1.0),
                            )
                        )

        # ── Post position bias ───────────────────────────────
        for zone in ["inside", "middle", "outside"]:
            p = post_stats.get(zone, {"wins": 0, "top4": 0, "total": 0})
            if p["total"] >= self.MIN_MILD:
                win_pct = p["wins"] / p["total"] * 100
                top4_pct = p["top4"] / p["total"] * 100
                if win_pct >= 40:
                    insights.append(
                        BiasInsight(
                            category="post",
                            severity="strong"
                            if p["total"] >= self.MIN_CONFIDENT
                            else "moderate",
                            description=f"{zone.title()} posts ({zone} 1/3 of field) win {win_pct:.0f}% — Rail/Post bias detected",
                            stat=win_pct,
                            sample_size=p["total"],
                            confidence=min(p["total"] / 50, 1.0),
                        )
                    )
                if top4_pct >= 70:
                    insights.append(
                        BiasInsight(
                            category="post",
                            severity="moderate",
                            description=f"{zone.title()} posts hit top 4 at {top4_pct:.0f}% rate",
                            stat=top4_pct,
                            sample_size=p["total"],
                            confidence=min(p["total"] / 50, 1.0),
                        )
                    )

        # ── Surface accuracy divergence ──────────────────────
        if profile.dirt_races >= self.MIN_MILD and profile.turf_races >= self.MIN_MILD:
            diff = abs(profile.dirt_winner_pct - profile.turf_winner_pct)
            if diff >= 15:
                better = (
                    "Dirt"
                    if profile.dirt_winner_pct > profile.turf_winner_pct
                    else "Turf"
                )
                insights.append(
                    BiasInsight(
                        category="surface",
                        severity="moderate",
                        description=f"Model is {diff:.0f}% more accurate on {better} — consider surface-specific tuning",
                        stat=diff,
                        sample_size=min(profile.dirt_races, profile.turf_races),
                        confidence=0.7,
                    )
                )

        # ── Distance accuracy divergence ─────────────────────
        if (
            profile.sprint_races >= self.MIN_MILD
            and profile.route_races >= self.MIN_MILD
        ):
            diff = abs(profile.sprint_winner_pct - profile.route_winner_pct)
            if diff >= 15:
                better = (
                    "Sprint"
                    if profile.sprint_winner_pct > profile.route_winner_pct
                    else "Route"
                )
                insights.append(
                    BiasInsight(
                        category="distance",
                        severity="moderate",
                        description=f"Model is {diff:.0f}% more accurate on {better} distances",
                        stat=diff,
                        sample_size=min(profile.sprint_races, profile.route_races),
                        confidence=0.7,
                    )
                )

        return insights

    def save_profile(
        self,
        profile: TrackProfile,
        surface: str = "all",
        distance_bucket: str = "all",
        condition_bucket: str = "all",
    ):
        """Persist a track profile to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            profile_data = {
                "total_races": profile.total_races,
                "total_horses": profile.total_horses,
                "dirt_races": profile.dirt_races,
                "dirt_winner_pct": profile.dirt_winner_pct,
                "turf_races": profile.turf_races,
                "turf_winner_pct": profile.turf_winner_pct,
                "sprint_races": profile.sprint_races,
                "sprint_winner_pct": profile.sprint_winner_pct,
                "route_races": profile.route_races,
                "route_winner_pct": profile.route_winner_pct,
                "fast_races": profile.fast_races,
                "fast_winner_pct": profile.fast_winner_pct,
                "off_track_races": profile.off_track_races,
                "off_track_winner_pct": profile.off_track_winner_pct,
                "style_bias": profile.style_bias,
                "speed_win_pct": profile.speed_win_pct,
                "presser_win_pct": profile.presser_win_pct,
                "closer_win_pct": profile.closer_win_pct,
                "inside_win_pct": profile.inside_win_pct,
                "inside_top4_pct": profile.inside_top4_pct,
                "middle_win_pct": profile.middle_win_pct,
                "middle_top4_pct": profile.middle_top4_pct,
                "outside_win_pct": profile.outside_win_pct,
                "outside_top4_pct": profile.outside_top4_pct,
                "overall_winner_pct": profile.overall_winner_pct,
                "overall_top3_pct": profile.overall_top3_pct,
                "overall_top4_pct": profile.overall_top4_pct,
                "top_jt_combos": profile.top_jt_combos,
            }

            insights_data = [
                {
                    "category": ins.category,
                    "severity": ins.severity,
                    "description": ins.description,
                    "stat": ins.stat,
                    "sample_size": ins.sample_size,
                    "confidence": ins.confidence,
                }
                for ins in profile.insights
            ]

            cursor.execute(
                """
                INSERT INTO track_intelligence_profiles
                    (track_code, surface, distance_bucket, condition_bucket,
                     profile_json, insights_json, total_races, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_code, surface, distance_bucket, condition_bucket)
                DO UPDATE SET
                    profile_json = excluded.profile_json,
                    insights_json = excluded.insights_json,
                    total_races = excluded.total_races,
                    last_updated = excluded.last_updated
            """,
                (
                    track_code,
                    surface,
                    distance_bucket,
                    condition_bucket,
                    json.dumps(profile_data),
                    json.dumps(insights_data),
                    profile.total_races,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
            logger.info(
                f"Track intelligence saved: {track_code} ({profile.total_races} races)"
            )

        except Exception as e:
            logger.warning(f"Failed to save track intelligence: {e}")

    def load_profile(
        self,
        track_code: str,
        surface: str = "all",
        distance_bucket: str = "all",
        condition_bucket: str = "all",
    ) -> TrackProfile | None:
        """Load a saved track profile from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT profile_json, insights_json, total_races
                FROM track_intelligence_profiles
                WHERE UPPER(track_code) = UPPER(?)
                  AND surface = ? AND distance_bucket = ? AND condition_bucket = ?
            """,
                (track_code, surface, distance_bucket, condition_bucket),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            data = json.loads(row[0])
            profile = TrackProfile(track_code=track_code)
            for k, v in data.items():
                if hasattr(profile, k) and k != "top_jt_combos":
                    setattr(profile, k, v)
                elif k == "top_jt_combos":
                    profile.top_jt_combos = v

            # Rebuild insight objects
            if row[1]:
                insights_raw = json.loads(row[1])
                profile.insights = [BiasInsight(**ins) for ins in insights_raw]

            return profile

        except Exception as e:
            logger.warning(f"Failed to load track profile: {e}")
            return None

    def rebuild_all_profiles(self):
        """Rebuild intelligence profiles for all tracks in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT UPPER(track) FROM gold_high_iq")
            tracks = [r[0] for r in cursor.fetchall()]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to list tracks: {e}")
            return

        for track in tracks:
            profile = self.build_full_profile(track)
            self.save_profile(profile)
            logger.info(
                f"  {track}: {profile.total_races} races, "
                f"{len(profile.insights)} insights detected"
            )

    def update_after_submission(self, track_code: str):
        """
        Rebuild just one track's profile after a new result submission.
        Called from the Submit Top 4 flow.
        """
        profile = self.build_full_profile(track_code)
        self.save_profile(profile)
        return profile

    def get_all_track_summaries(self) -> list[dict]:
        """Get summary data for all tracks (for dashboard overview)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT track_code, profile_json, insights_json, total_races, last_updated
                FROM track_intelligence_profiles
                WHERE surface = 'all' AND distance_bucket = 'all' AND condition_bucket = 'all'
                ORDER BY total_races DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            summaries = []
            for row in rows:
                data = json.loads(row[1])
                insights = json.loads(row[2]) if row[2] else []
                summaries.append(
                    {
                        "track": row[0],
                        "total_races": row[3],
                        "last_updated": row[4],
                        "overall_winner_pct": data.get("overall_winner_pct", 0),
                        "overall_top3_pct": data.get("overall_top3_pct", 0),
                        "overall_top4_pct": data.get("overall_top4_pct", 0),
                        "style_bias": data.get("style_bias", "Unknown"),
                        "n_insights": len(insights),
                        "top_insight": insights[0]["description"]
                        if insights
                        else "No biases detected yet",
                    }
                )
            return summaries

        except Exception as e:
            logger.warning(f"Failed to get track summaries: {e}")
            return []

    def save_track_ml_profile(
        self,
        track_code: str,
        model_path: str,
        n_features: int,
        hidden_dim: int,
        val_metrics: dict,
        races_trained: int,
        blend_weight: float = 1.5,
    ):
        """Save per-track ML model metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO track_ml_profiles
                    (track_code, model_path, n_features, hidden_dim,
                     val_winner_acc, val_top3_acc, val_top4_acc,
                     races_trained, last_retrained, blend_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_code) DO UPDATE SET
                    model_path = excluded.model_path,
                    n_features = excluded.n_features,
                    hidden_dim = excluded.hidden_dim,
                    val_winner_acc = excluded.val_winner_acc,
                    val_top3_acc = excluded.val_top3_acc,
                    val_top4_acc = excluded.val_top4_acc,
                    races_trained = excluded.races_trained,
                    last_retrained = excluded.last_retrained,
                    blend_weight = excluded.blend_weight
            """,
                (
                    track_code.upper(),
                    model_path,
                    n_features,
                    hidden_dim,
                    val_metrics.get("winner_accuracy", 0),
                    val_metrics.get("top3_accuracy", 0),
                    val_metrics.get("top4_accuracy", 0),
                    races_trained,
                    datetime.now().isoformat(),
                    blend_weight,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save track ML profile: {e}")

    def get_track_ml_profile(self, track_code: str) -> dict | None:
        """Get per-track ML model metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_path, n_features, hidden_dim,
                       val_winner_acc, val_top3_acc, val_top4_acc,
                       races_trained, last_retrained, blend_weight
                FROM track_ml_profiles
                WHERE UPPER(track_code) = UPPER(?)
            """,
                (track_code,),
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "model_path": row[0],
                "n_features": row[1],
                "hidden_dim": row[2],
                "val_winner_acc": row[3],
                "val_top3_acc": row[4],
                "val_top4_acc": row[5],
                "races_trained": row[6],
                "last_retrained": row[7],
                "blend_weight": row[8],
            }
        except Exception:
            return None
