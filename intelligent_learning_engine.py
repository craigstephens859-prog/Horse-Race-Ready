"""
INTELLIGENT LEARNING ENGINE
============================

Analyzes race results using the same high-IQ logic from manual training sessions.
Identifies WHY predictions were wrong and learns specific patterns.

Key Learnings from Feb 5, 2026 TUP R5 Training Session:
1. Best Recent Speed - Winner had highest last race speed but was underrated
2. Class Drop Pattern - Horses dropping in class with speed often win
3. Layoff Cycle Bounce - 3rd/4th start off layoff with improvement
4. C-Form/Speed Override - Don't let declining form kill strong recent speed
5. Lone Presser in Hot Pace - Only P style in hot pace = value spot
6. Post Bias Alignment - Match post ratings to actual track bias data

Author: Elite ML Engineer
Date: February 5, 2026
"""

import sqlite3
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RaceInsight:
    """A learning insight from comparing predictions to actual results."""
    pattern_type: str  # e.g., 'best_recent_speed', 'class_drop', 'lone_presser'
    description: str
    predicted_rank: int
    actual_rank: int
    horse_name: str
    confidence: float  # 0-1 how confident we are this pattern explains the miss
    suggested_adjustment: str  # Human-readable adjustment suggestion
    weight_key: str  # Key for auto-calibration
    weight_adjustment: float  # How much to adjust (positive = increase importance)


class IntelligentLearningEngine:
    """
    Analyzes race results using high-IQ pattern recognition.
    Goes beyond simple gradient descent to understand WHY predictions failed.
    """

    def __init__(self, db_path: str = "gold_high_iq.db"):
        self.db_path = db_path
        self._init_learning_tables()
        
        # Track pattern frequency for meta-learning
        self.pattern_counts = self._load_pattern_history()

    def _init_learning_tables(self):
        """Create tables to store intelligent learning insights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for race insights
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS race_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT,
                predicted_rank INTEGER,
                actual_rank INTEGER,
                horse_name TEXT,
                confidence REAL,
                weight_key TEXT,
                weight_adjustment REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for pattern frequency tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_frequency (
                pattern_type TEXT PRIMARY KEY,
                occurrence_count INTEGER DEFAULT 0,
                avg_rank_improvement REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for learned feature flag adjustments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_feature_flags (
                flag_name TEXT PRIMARY KEY,
                should_enable BOOLEAN DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                supporting_races INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Intelligent learning tables initialized")

    def _load_pattern_history(self) -> Dict[str, int]:
        """Load historical pattern frequency."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT pattern_type, occurrence_count FROM pattern_frequency")
            rows = cursor.fetchall()
            conn.close()
            return {row[0]: row[1] for row in rows}
        except:
            return {}

    def analyze_race_result(
        self,
        race_id: str,
        predictions: List[Dict],
        actual_results: List[int],  # List of program numbers in finish order
        pp_data: Optional[Dict] = None,  # Brisnet PP data if available
        track_bias: Optional[Dict] = None  # Track bias data if available
    ) -> List[RaceInsight]:
        """
        Analyze a race result and identify why predictions were wrong.
        
        Args:
            race_id: Unique race identifier
            predictions: List of horse dicts with predicted_rank, rating, etc.
            actual_results: Program numbers in finish order [1st, 2nd, 3rd, 4th, 5th]
            pp_data: Optional Brisnet PP data for deeper analysis
            track_bias: Optional track bias data
            
        Returns:
            List of RaceInsight objects explaining prediction errors
        """
        insights = []
        
        # Build lookup maps
        prog_to_pred = {h.get('program_number', h.get('post', i+1)): h 
                        for i, h in enumerate(predictions)}
        
        # Map actual finish to program numbers
        actual_map = {prog: rank+1 for rank, prog in enumerate(actual_results[:5])}
        
        # Get winner and top finishers
        winner_prog = actual_results[0] if actual_results else None
        winner_data = prog_to_pred.get(winner_prog, {})
        winner_predicted_rank = winner_data.get('predicted_rank', 99)
        
        # Only analyze if we missed the winner significantly
        if winner_predicted_rank > 3:
            # === PATTERN 1: BEST RECENT SPEED ===
            insights.extend(self._check_best_speed_pattern(
                predictions, actual_results, prog_to_pred
            ))
            
            # === PATTERN 2: CLASS DROP ===
            insights.extend(self._check_class_drop_pattern(
                predictions, actual_results, prog_to_pred
            ))
            
            # === PATTERN 3: LAYOFF CYCLE BOUNCE ===
            insights.extend(self._check_layoff_cycle_pattern(
                predictions, actual_results, prog_to_pred
            ))
            
            # === PATTERN 4: LONE PRESSER IN HOT PACE ===
            insights.extend(self._check_lone_presser_pattern(
                predictions, actual_results, prog_to_pred
            ))
            
            # === PATTERN 5: TRACK BIAS ALIGNMENT ===
            if track_bias:
                insights.extend(self._check_track_bias_pattern(
                    predictions, actual_results, prog_to_pred, track_bias
                ))
            
            # === PATTERN 6: C-FORM / SPEED OVERRIDE ===
            insights.extend(self._check_form_speed_override_pattern(
                predictions, actual_results, prog_to_pred
            ))
        
        # Store insights in database
        self._save_insights(race_id, insights)
        
        return insights

    def _check_best_speed_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict
    ) -> List[RaceInsight]:
        """Check if winner had best recent speed but was underrated."""
        insights = []
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        
        # Get all last race speeds
        speeds = []
        for prog, data in prog_to_pred.items():
            last_fig = data.get('last_fig') or data.get('speed_last', 0)
            if last_fig and last_fig > 0:
                speeds.append((prog, last_fig))
        
        if not speeds:
            return insights
        
        # Sort by speed descending
        speeds_sorted = sorted(speeds, key=lambda x: x[1], reverse=True)
        
        # Did winner have best or top-3 speed?
        winner_speed_rank = None
        for rank, (prog, spd) in enumerate(speeds_sorted, 1):
            if prog == winner_prog:
                winner_speed_rank = rank
                winner_speed = spd
                break
        
        if winner_speed_rank and winner_speed_rank <= 3:
            winner_predicted_rank = winner_data.get('predicted_rank', 99)
            
            if winner_predicted_rank > 5:  # We missed this significantly
                insights.append(RaceInsight(
                    pattern_type='best_recent_speed',
                    description=f"Winner had #{winner_speed_rank} best recent speed ({winner_speed}) but was ranked #{winner_predicted_rank}",
                    predicted_rank=winner_predicted_rank,
                    actual_rank=1,
                    horse_name=winner_data.get('horse_name', 'Unknown'),
                    confidence=0.85,
                    suggested_adjustment="Increase weight for horses with best recent speed in field",
                    weight_key='last_race_speed_bonus',
                    weight_adjustment=0.1
                ))
        
        return insights

    def _check_class_drop_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict
    ) -> List[RaceInsight]:
        """Check if winner was dropping in class."""
        insights = []
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        
        # Check for class drop indicators
        # Look at rating components if available
        c_class = winner_data.get('c_class', winner_data.get('Cclass', 0))
        recent_purses = winner_data.get('recent_purses', [])
        
        # Also check if "drop" is mentioned in any angles
        angles = winner_data.get('angles', [])
        has_drop_angle = any('drop' in str(a).lower() or 'class' in str(a).lower() 
                            for a in angles) if angles else False
        
        # Positive c_class often indicates horse is dropping
        winner_predicted_rank = winner_data.get('predicted_rank', 99)
        
        if (c_class > 0.5 or has_drop_angle) and winner_predicted_rank > 4:
            insights.append(RaceInsight(
                pattern_type='class_drop',
                description=f"Winner was dropping in class (c_class={c_class:.2f}) but ranked #{winner_predicted_rank}",
                predicted_rank=winner_predicted_rank,
                actual_rank=1,
                horse_name=winner_data.get('horse_name', 'Unknown'),
                confidence=0.75,
                suggested_adjustment="Increase bonus for horses dropping in class with decent speed",
                weight_key='class_drop_bonus',
                weight_adjustment=0.15
            ))
        
        return insights

    def _check_layoff_cycle_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict
    ) -> List[RaceInsight]:
        """Check if winner was in 3rd/4th start off layoff with improving figures."""
        insights = []
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        
        # Check form trend
        c_form = winner_data.get('c_form', winner_data.get('Cform', 0))
        speed_figs = winner_data.get('speed_figures', [])
        
        # Positive c_form with recent figures suggests improvement pattern
        winner_predicted_rank = winner_data.get('predicted_rank', 99)
        
        if c_form > 0 and winner_predicted_rank > 5:
            # Check if improving trend in figures
            if len(speed_figs) >= 3:
                recent_avg = np.mean(speed_figs[:2]) if speed_figs[:2] else 0
                older_avg = np.mean(speed_figs[2:4]) if len(speed_figs) > 2 else 0
                
                if recent_avg > older_avg:  # Improving
                    insights.append(RaceInsight(
                        pattern_type='layoff_cycle_bounce',
                        description=f"Winner was improving off layoff (c_form={c_form:.2f}, figs trending up) but ranked #{winner_predicted_rank}",
                        predicted_rank=winner_predicted_rank,
                        actual_rank=1,
                        horse_name=winner_data.get('horse_name', 'Unknown'),
                        confidence=0.70,
                        suggested_adjustment="Add bonus for 3rd/4th start off layoff with improving figures",
                        weight_key='layoff_cycle_bonus',
                        weight_adjustment=0.1
                    ))
        
        return insights

    def _check_lone_presser_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict
    ) -> List[RaceInsight]:
        """Check if winner was the only Presser (P) style in a hot pace."""
        insights = []
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        winner_style = winner_data.get('pace_style', winner_data.get('Pace_Style', ''))
        
        if winner_style != 'P':
            return insights
        
        # Count pace styles in field
        style_counts = {'E': 0, 'E/P': 0, 'P': 0, 'S': 0}
        for data in prog_to_pred.values():
            style = data.get('pace_style', data.get('Pace_Style', ''))
            if style in style_counts:
                style_counts[style] += 1
        
        # Is winner the only P?
        if style_counts['P'] == 1:
            early_speed = style_counts['E'] + style_counts['E/P']
            winner_predicted_rank = winner_data.get('predicted_rank', 99)
            
            if early_speed >= 2 and winner_predicted_rank > 4:  # Hot pace scenario
                insights.append(RaceInsight(
                    pattern_type='lone_presser_hot_pace',
                    description=f"Winner was lone Presser in hot pace ({early_speed} E/E/P types) but ranked #{winner_predicted_rank}",
                    predicted_rank=winner_predicted_rank,
                    actual_rank=1,
                    horse_name=winner_data.get('horse_name', 'Unknown'),
                    confidence=0.80,
                    suggested_adjustment="Reduce P style penalty when lone presser in hot pace",
                    weight_key='lone_presser_adjustment',
                    weight_adjustment=0.15
                ))
        
        return insights

    def _check_track_bias_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict,
        track_bias: Dict
    ) -> List[RaceInsight]:
        """Check if track bias was not properly weighted."""
        insights = []
        
        # Extract bias info
        post_bias = track_bias.get('post_bias', '')
        style_bias = track_bias.get('style_bias', [])
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        winner_post = int(winner_data.get('post', winner_data.get('Post', 0)))
        winner_style = winner_data.get('pace_style', winner_data.get('Pace_Style', ''))
        winner_predicted_rank = winner_data.get('predicted_rank', 99)
        
        # Check post alignment
        if '4-7' in str(post_bias) or 'mid' in str(post_bias).lower():
            if 4 <= winner_post <= 7 and winner_predicted_rank > 5:
                insights.append(RaceInsight(
                    pattern_type='post_bias_alignment',
                    description=f"Winner from favored mid-post ({winner_post}) but ranked #{winner_predicted_rank}",
                    predicted_rank=winner_predicted_rank,
                    actual_rank=1,
                    horse_name=winner_data.get('horse_name', 'Unknown'),
                    confidence=0.65,
                    suggested_adjustment="Better align post ratings with track bias data",
                    weight_key='track_bias_post_alignment',
                    weight_adjustment=0.1
                ))
        
        return insights

    def _check_form_speed_override_pattern(
        self,
        predictions: List[Dict],
        actual_results: List[int],
        prog_to_pred: Dict
    ) -> List[RaceInsight]:
        """Check if winner had strong recent speed but was penalized for declining form."""
        insights = []
        
        winner_prog = actual_results[0]
        winner_data = prog_to_pred.get(winner_prog, {})
        
        c_form = winner_data.get('c_form', winner_data.get('Cform', 0))
        last_speed = winner_data.get('last_fig') or winner_data.get('speed_last', 0)
        winner_predicted_rank = winner_data.get('predicted_rank', 99)
        
        # Low form but strong recent speed
        if c_form < 0.3 and last_speed >= 80 and winner_predicted_rank > 5:
            insights.append(RaceInsight(
                pattern_type='form_speed_override',
                description=f"Winner had strong speed ({last_speed}) but low form ({c_form:.2f}), ranked #{winner_predicted_rank}",
                predicted_rank=winner_predicted_rank,
                actual_rank=1,
                horse_name=winner_data.get('horse_name', 'Unknown'),
                confidence=0.75,
                suggested_adjustment="Don't penalize form heavily when recent speed is strong (>=80)",
                weight_key='cform_speed_override',
                weight_adjustment=0.1
            ))
        
        return insights

    def _save_insights(self, race_id: str, insights: List[RaceInsight]):
        """Save insights to database for future reference."""
        if not insights:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for insight in insights:
                cursor.execute("""
                    INSERT INTO race_insights 
                    (race_id, pattern_type, description, predicted_rank, actual_rank,
                     horse_name, confidence, weight_key, weight_adjustment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    insight.pattern_type,
                    insight.description,
                    insight.predicted_rank,
                    insight.actual_rank,
                    insight.horse_name,
                    insight.confidence,
                    insight.weight_key,
                    insight.weight_adjustment
                ))
                
                # Update pattern frequency
                cursor.execute("""
                    INSERT INTO pattern_frequency (pattern_type, occurrence_count, avg_rank_improvement)
                    VALUES (?, 1, ?)
                    ON CONFLICT(pattern_type) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        avg_rank_improvement = (avg_rank_improvement * occurrence_count + ?) / (occurrence_count + 1),
                        last_updated = CURRENT_TIMESTAMP
                """, (
                    insight.pattern_type,
                    insight.predicted_rank - insight.actual_rank,
                    insight.predicted_rank - insight.actual_rank
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Saved {len(insights)} insights for race {race_id}")
            
        except Exception as e:
            logger.warning(f"Could not save insights: {e}")

    def get_accumulated_learnings(self) -> Dict:
        """Get summary of all learnings to apply to the model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get pattern frequency and effectiveness
            cursor.execute("""
                SELECT pattern_type, occurrence_count, avg_rank_improvement
                FROM pattern_frequency
                ORDER BY occurrence_count DESC
            """)
            patterns = cursor.fetchall()
            
            # Get recent insights
            cursor.execute("""
                SELECT pattern_type, weight_key, AVG(weight_adjustment) as avg_adj,
                       COUNT(*) as count, AVG(confidence) as avg_conf
                FROM race_insights
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY pattern_type, weight_key
                HAVING count >= 3
                ORDER BY count DESC
            """)
            recent_insights = cursor.fetchall()
            
            conn.close()
            
            # Build recommendations
            recommendations = []
            for pattern, weight_key, avg_adj, count, avg_conf in recent_insights:
                if avg_conf >= 0.65:  # Only high confidence patterns
                    recommendations.append({
                        'pattern': pattern,
                        'weight_key': weight_key,
                        'suggested_adjustment': round(avg_adj, 3),
                        'supporting_races': count,
                        'confidence': round(avg_conf, 2)
                    })
            
            return {
                'pattern_frequency': {p[0]: {'count': p[1], 'avg_improvement': p[2]} 
                                     for p in patterns},
                'recommendations': recommendations,
                'total_races_analyzed': sum(p[1] for p in patterns) if patterns else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not get learnings: {e}")
            return {'pattern_frequency': {}, 'recommendations': [], 'total_races_analyzed': 0}

    def apply_learnings_to_feature_flags(self) -> Dict[str, bool]:
        """
        Based on accumulated learnings, determine which feature flags should be enabled.
        
        Returns:
            Dict of feature flag names to recommended enabled state
        """
        learnings = self.get_accumulated_learnings()
        
        # Map patterns to feature flags
        pattern_to_flag = {
            'best_recent_speed': 'use_last_race_speed_bonus',
            'class_drop': 'use_class_drop_bonus',
            'layoff_cycle_bounce': 'use_layoff_cycle_bonus',
            'form_speed_override': 'use_cform_speed_override',
            'lone_presser_hot_pace': 'use_lone_presser_adjustment',
            'post_bias_alignment': 'use_track_bias_post_alignment'
        }
        
        recommendations = {}
        for pattern, freq_data in learnings.get('pattern_frequency', {}).items():
            if pattern in pattern_to_flag:
                flag = pattern_to_flag[pattern]
                # Enable if pattern has been seen 3+ times with positive improvement
                should_enable = freq_data['count'] >= 3 and freq_data['avg_improvement'] > 0
                recommendations[flag] = should_enable
        
        return recommendations


def analyze_and_learn_from_result(
    db_path: str,
    race_id: str,
    predictions: List[Dict],
    actual_results: List[int],
    pp_data: Optional[Dict] = None,
    track_bias: Optional[Dict] = None
) -> Dict:
    """
    Main entry point for intelligent learning after result submission.
    
    Called from app.py after user submits actual results.
    
    Returns:
        Dict with insights and recommendations
    """
    engine = IntelligentLearningEngine(db_path)
    
    # Analyze the race
    insights = engine.analyze_race_result(
        race_id=race_id,
        predictions=predictions,
        actual_results=actual_results,
        pp_data=pp_data,
        track_bias=track_bias
    )
    
    # Get accumulated learnings
    learnings = engine.get_accumulated_learnings()
    
    # Get feature flag recommendations
    flag_recommendations = engine.apply_learnings_to_feature_flags()
    
    return {
        'race_id': race_id,
        'insights_found': len(insights),
        'insights': [
            {
                'pattern': i.pattern_type,
                'description': i.description,
                'horse': i.horse_name,
                'confidence': i.confidence
            }
            for i in insights
        ],
        'total_patterns_learned': learnings.get('total_races_analyzed', 0),
        'top_patterns': list(learnings.get('pattern_frequency', {}).keys())[:5],
        'feature_flag_recommendations': flag_recommendations,
        'weight_recommendations': learnings.get('recommendations', [])
    }


if __name__ == "__main__":
    # Test the engine
    print("Testing Intelligent Learning Engine...")
    
    # Simulate a race where model missed
    test_predictions = [
        {'program_number': 9, 'horse_name': 'Bendettijoe', 'predicted_rank': 1, 'c_form': 0.81, 'pace_style': 'E/P'},
        {'program_number': 2, 'horse_name': 'Outofquemado', 'predicted_rank': 2, 'c_form': 0.52, 'pace_style': 'S'},
        {'program_number': 3, 'horse_name': 'Enos Slaughter', 'predicted_rank': 8, 'c_form': 0.11, 'pace_style': 'P', 'last_fig': 82, 'c_class': 0.8},
        {'program_number': 7, 'horse_name': 'Silver Dash', 'predicted_rank': 7, 'pace_style': 'E/P'},
        {'program_number': 1, 'horse_name': 'Hadlees Honor', 'predicted_rank': 4, 'pace_style': 'E/P'},
    ]
    
    # Actual results: 3, 7, 1, 6, 2
    test_actual = [3, 7, 1, 6, 2]
    
    result = analyze_and_learn_from_result(
        db_path="gold_high_iq.db",
        race_id="TEST_TUP_R5_20260205",
        predictions=test_predictions,
        actual_results=test_actual
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"   Insights found: {result['insights_found']}")
    for insight in result['insights']:
        print(f"   - {insight['pattern']}: {insight['description']}")
    print(f"\n   Feature flag recommendations: {result['feature_flag_recommendations']}")
