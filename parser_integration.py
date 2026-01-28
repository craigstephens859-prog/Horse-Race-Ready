#!/usr/bin/env python3
"""
🔗 ELITE PARSER → RATING ENGINE INTEGRATION
Bridges parsed PP data to probabilistic predictions with 90% accuracy target.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from elite_parser import EliteBRISNETParser, HorseData

# ===================== INTEGRATION LAYER =====================

class ParserToRatingBridge:
    """
    Converts parsed horse data → rating calculation → softmax probabilities.
    Ensures seamless flow: PP text → structured dict → ratings → predictions.
    """
    
    def __init__(self, softmax_tau: float = 3.0):
        self.parser = EliteBRISNETParser()
        self.softmax_tau = softmax_tau
        self.last_parse_report = None
    
    def parse_and_rate(self, pp_text: str, 
                      today_purse: int,
                      today_race_type: str,
                      track_name: str,
                      surface_type: str,
                      distance_txt: str) -> pd.DataFrame:
        """
        End-to-end pipeline: PP text → predictions
        
        Returns DataFrame with columns:
            Horse, Post, Rating, Probability, Fair_Odds, Confidence
        """
        
        # Step 1: Parse PP text
        print("🔍 Parsing PP text...")
        horses = self.parser.parse_full_pp(pp_text)
        
        if not horses:
            raise ValueError("❌ No horses could be parsed from PP text")
        
        print(f"✅ Parsed {len(horses)} horses")
        
        # Step 2: Validate parsing quality
        validation = self.parser.validate_parsed_data(horses)
        self.last_parse_report = validation
        
        if validation['overall_confidence'] < 0.7:
            print(f"⚠️ WARNING: Low parsing confidence ({validation['overall_confidence']:.1%})")
            print(f"   Issues: {len(validation['issues'])}")
            for issue in validation['critical_issues']:
                print(f"      - {issue}")
        
        # Step 3: Calculate ratings for each horse
        print("\n📊 Calculating ratings...")
        rows = []
        
        for name, horse in horses.items():
            rating = self._calculate_comprehensive_rating(
                horse, today_purse, today_race_type, 
                track_name, surface_type, distance_txt
            )
            
            rows.append({
                'Horse': name,
                'Post': horse.post,
                'Rating': rating,
                'Pace_Style': horse.pace_style,
                'Quirin': horse.quirin_points,
                'ML_Odds': horse.ml_odds,
                'Jockey': horse.jockey,
                'Trainer': horse.trainer,
                'Speed_Figs': f"{horse.avg_top2:.1f}" if horse.speed_figures else "N/A",
                'Angles': horse.angle_count,
                'Parse_Confidence': horse.parsing_confidence
            })
        
        df = pd.DataFrame(rows)
        
        # Step 4: Apply softmax to get probabilities
        print("🎲 Computing probabilities...")
        df = self._apply_softmax(df)
        
        # Step 5: Calculate fair odds
        df['Fair_Odds'] = (1.0 / df['Probability']).round(2)
        df['Fair_Odds_AM'] = df['Probability'].apply(self._prob_to_american)
        
        # Sort by probability descending
        df = df.sort_values('Probability', ascending=False).reset_index(drop=True)
        df['Predicted_Finish'] = df.index + 1
        
        return df
    
    def _calculate_comprehensive_rating(self, 
                                       horse: HorseData,
                                       today_purse: int,
                                       today_race_type: str,
                                       track_name: str,
                                       surface_type: str,
                                       distance_txt: str) -> float:
        """
        Calculate final rating using all parsed data.
        Mirrors app.py rating formula but uses parsed HorseData.
        
        Rating = (Cclass×2.5) + (Cform×1.8) + (Cspeed×2.0) + 
                 (Cpace×1.5) + (Cstyle×1.2) + (Cpost×0.8) + 
                 Angles_Bonus
        """
        
        # Component 1: CLASS (Cclass) [-3.0 to +6.0]
        cclass = self._calculate_class_rating(horse, today_purse, today_race_type)
        
        # Component 2: FORM CYCLE (Cform) [-3.0 to +3.0]
        cform = self._calculate_form_rating(horse)
        
        # Component 3: SPEED (Cspeed) [-2.0 to +2.0]
        cspeed = self._calculate_speed_rating(horse, horses_in_race=None)  # Would pass all horses
        
        # Component 4: PACE (Cpace) [-3.0 to +3.0]
        cpace = self._calculate_pace_rating(horse, distance_txt)
        
        # Component 5: STYLE (Cstyle) [-0.5 to +0.8]
        cstyle = self._calculate_style_rating(horse, surface_type)
        
        # Component 6: POST POSITION (Cpost) [-0.5 to +0.5]
        cpost = self._calculate_post_rating(horse, distance_txt)
        
        # Component 7: ANGLES BONUS
        angles_bonus = horse.angle_count * 0.10
        
        # Weighted formula
        rating = (
            (cclass * 2.5) +
            (cform * 1.8) +
            (cspeed * 2.0) +
            (cpace * 1.5) +
            (cstyle * 1.2) +
            (cpost * 0.8) +
            angles_bonus
        )
        
        return round(rating, 2)
    
    def _calculate_class_rating(self, horse: HorseData, today_purse: int, today_race_type: str) -> float:
        """Class rating based on purse movement and race type hierarchy"""
        
        race_type_scores = {
            'mcl': 1, 'clm': 2, 'mdn': 3, 'md sp wt': 3, 'msw': 3,
            'alw': 4, 'oc': 4.5, 'stk': 5, 'hcp': 5.5,
            'g3': 6, 'g2': 7, 'g1': 8
        }
        
        today_score = race_type_scores.get(today_race_type.lower(), 3.5)
        rating = 0.0
        
        # Purse comparison
        if horse.recent_purses and today_purse > 0:
            avg_recent = np.mean(horse.recent_purses)
            purse_ratio = today_purse / avg_recent if avg_recent > 0 else 1.0
            
            if purse_ratio >= 1.5:  # Major step up
                rating -= 1.2
            elif purse_ratio >= 1.2:
                rating -= 0.6
            elif 0.8 <= purse_ratio <= 1.2:  # Same class
                rating += 0.8
            elif purse_ratio >= 0.6:  # Class drop
                rating += 1.5
            else:  # Major drop
                rating += 2.5
        
        # Race type progression
        if horse.race_types:
            recent_scores = [race_type_scores.get(rt.lower(), 3.5) for rt in horse.race_types]
            avg_recent_type = np.mean(recent_scores)
            type_diff = today_score - avg_recent_type
            
            if type_diff >= 2.0:
                rating -= 1.5
            elif type_diff >= 1.0:
                rating -= 0.8
            elif abs(type_diff) < 0.5:
                rating += 0.5
            elif type_diff <= -1.0:
                rating += 1.2
        
        # Pedigree quality boost
        if horse.sire_spi and horse.sire_spi >= 110:
            rating += 0.4
        elif horse.sire_spi and horse.sire_spi >= 100:
            rating += 0.2
        
        # Absolute purse baseline
        if today_purse >= 100000:
            rating += 1.0
        elif today_purse >= 50000:
            rating += 0.5
        elif today_purse < 25000:
            rating -= 0.5
        
        return float(np.clip(rating, -3.0, 6.0))
    
    def _calculate_form_rating(self, horse: HorseData) -> float:
        """Form cycle rating based on layoff, trend, recent wins"""
        
        # First-time starter special handling
        if not horse.speed_figures and not horse.recent_finishes:
            # Use pedigree and angles to evaluate debut horse
            debut_rating = 0.0
            
            # Pedigree quality
            if horse.sire_spi:
                if horse.sire_spi >= 115:
                    debut_rating += 1.2
                elif horse.sire_spi >= 110:
                    debut_rating += 0.8
            
            # Trainer debut angles
            debut_angles = [a for a in horse.angles if 'debut' in a['category'].lower() or '1st time' in a['category'].lower()]
            if debut_angles:
                for angle in debut_angles:
                    if angle['roi'] > 1.0:
                        debut_rating += 0.8
                    elif angle['roi'] > 0.0:
                        debut_rating += 0.4
            
            return float(np.clip(debut_rating, -2.0, 3.0))
        
        rating = 0.0
        
        # Layoff factor
        if horse.days_since_last is not None:
            days = horse.days_since_last
            if days <= 14:
                rating += 0.5  # Fresh
            elif days <= 30:
                rating += 0.3
            elif days <= 60:
                rating += 0.0
            elif days <= 90:
                rating -= 0.5
            elif days <= 180:
                rating -= 1.0
            else:
                rating -= 2.0  # Long layoff
        
        # Form trend (finish positions)
        if horse.recent_finishes and len(horse.recent_finishes) >= 3:
            finishes = horse.recent_finishes[:5]
            
            # Improving trend
            if finishes[0] < finishes[1] < finishes[2]:
                rating += 1.5
            elif finishes[0] < finishes[1]:
                rating += 0.8
            
            # Declining trend
            if finishes[0] > finishes[1] > finishes[2]:
                rating -= 1.2
            
            # Recent win bonus
            if finishes[0] == 1:
                rating += 0.8
            
            # Consistency bonus (all finishes ≤ 3rd)
            if all(f <= 3 for f in finishes):
                rating += 0.5
        
        return float(np.clip(rating, -3.0, 3.0))
    
    def _calculate_speed_rating(self, horse: HorseData, horses_in_race) -> float:
        """Speed figure rating relative to race average"""
        
        if not horse.speed_figures or horse.avg_top2 == 0:
            return 0.0  # Neutral for first-timers
        
        # Would calculate race average from all horses
        # For now, use simplified calculation
        # Assuming race average is ~85 for maiden races, ~95 for allowance
        
        # This should be: (horse_avg - race_avg) * 0.05
        # Placeholder: assume neutral comparison
        fig_differential = 0.0  # Would calculate properly with all horses
        
        return float(np.clip(fig_differential, -2.0, 2.0))
    
    def _calculate_pace_rating(self, horse: HorseData, distance_txt: str) -> float:
        """Pace scenario rating based on style and field composition"""
        
        # Simplified - would use full pace analysis
        # E/P in sprint = +0.5, S in route = +0.3, etc.
        
        is_sprint = 'furlong' in distance_txt.lower() or '6f' in distance_txt or '7f' in distance_txt
        
        if horse.pace_style in ("E", "E/P") and is_sprint:
            return 0.5
        elif horse.pace_style == "S" and not is_sprint:
            return 0.3
        else:
            return 0.0
    
    def _calculate_style_rating(self, horse: HorseData, surface_type: str) -> float:
        """Running style rating based on strength"""
        
        strength_values = {
            'Strong': 0.8,
            'Solid': 0.4,
            'Slight': 0.1,
            'Weak': -0.3
        }
        
        return strength_values.get(horse.style_strength, 0.0)
    
    def _calculate_post_rating(self, horse: HorseData, distance_txt: str) -> float:
        """Post position rating (inside better for sprints, outside for routes)"""
        
        try:
            post_num = int(horse.post.rstrip('A'))  # Handle 1A
        except:
            return 0.0
        
        is_sprint = 'furlong' in distance_txt.lower() or '6f' in distance_txt or '7f' in distance_txt
        
        if is_sprint:
            # Inside posts favored
            if post_num <= 3:
                return 0.3
            elif post_num >= 9:
                return -0.4
        else:
            # Routes: middle posts favored
            if 4 <= post_num <= 7:
                return 0.2
            elif post_num <= 2 or post_num >= 10:
                return -0.3
        
        return 0.0
    
    def _apply_softmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply softmax to ratings to get probabilities"""
        
        ratings = torch.tensor(df['Rating'].values, dtype=torch.float32)
        ratings_scaled = ratings / self.softmax_tau
        probs = torch.nn.functional.softmax(ratings_scaled, dim=0).numpy()
        
        df['Probability'] = probs
        
        return df
    
    def _prob_to_american(self, prob: float) -> str:
        """Convert probability to American odds format"""
        if prob <= 0:
            return "N/A"
        
        decimal = 1.0 / prob
        
        if decimal >= 2:
            american = (decimal - 1) * 100
            return f"+{int(american)}"
        else:
            american = -100 / (decimal - 1)
            return f"{int(american)}"
    
    def get_parse_report(self) -> dict:
        """Return last parsing validation report"""
        return self.last_parse_report or {}
    
    def generate_prediction_summary(self, df: pd.DataFrame) -> str:
        """Generate human-readable prediction summary"""
        
        summary = []
        summary.append("\n" + "="*80)
        summary.append("🎯 RUNNING ORDER PREDICTIONS (90% Accuracy Target)")
        summary.append("="*80)
        
        # Winner prediction
        winner = df.iloc[0]
        summary.append(f"\n🥇 WIN: #{winner['Post']} {winner['Horse']}")
        summary.append(f"   Probability: {winner['Probability']:.1%}")
        summary.append(f"   Fair Odds: {winner['Fair_Odds']:.2f} ({winner['Fair_Odds_AM']})")
        summary.append(f"   Rating: {winner['Rating']:.2f}")
        
        # Top contenders for 2nd
        second_contenders = df.iloc[1:3]
        summary.append(f"\n🥈 PLACE (2 contenders):")
        for _, horse in second_contenders.iterrows():
            summary.append(f"   #{horse['Post']} {horse['Horse']} - {horse['Probability']:.1%}")
        
        # Top contenders for 3rd/4th
        third_contenders = df.iloc[3:6]
        summary.append(f"\n🥉 SHOW (3 contenders):")
        for _, horse in third_contenders.iterrows():
            summary.append(f"   #{horse['Post']} {horse['Horse']} - {horse['Probability']:.1%}")
        
        # Parsing confidence
        avg_conf = df['Parse_Confidence'].mean()
        summary.append(f"\n📊 Data Quality:")
        summary.append(f"   Parsing Confidence: {avg_conf:.1%}")
        
        if avg_conf < 0.7:
            summary.append(f"   ⚠️ WARNING: Low parsing confidence - verify PP data")
        
        return "\n".join(summary)

# ===================== USAGE EXAMPLE =====================

if __name__ == "__main__":
    # Example usage
    sample_pp = """
Race 2 Mountaineer 'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025

1 Way of Appeal (S 3)
Own: Someone
7/2 Red
BARRIOS RICARDO (254 58-42-39 23%)
B. h. 3 (Mar)
Sire : Appeal (Not for Love) $25,000
Trnr: Cady Khalil (150 18-24-31 12%)
Prime Power: 101.5 (4th)
2025 456 15% 44% -0.68
JKYw/ Sprints 217 12% 39% -0.32
23Sep23 Mtn Md Sp Wt 16500 98 4th
15Aug23 Mtn Alw 25000 92 2nd

2 Fast Runner (E/P 7)
Own: Speed LLC
5/2 Blue
ORTIZ IRAD (456 112-89-67 25%)
B. g. 4
Trnr: Baffert Bob (892 215-178-132 24%)
2025 789 18% 45% +0.85
18Oct24 Mtn Md Sp Wt 16500 105 1st
"""
    
    bridge = ParserToRatingBridge(softmax_tau=3.0)
    
    try:
        predictions = bridge.parse_and_rate(
            pp_text=sample_pp,
            today_purse=16500,
            today_race_type='Md Sp Wt',
            track_name='Mountaineer',
            surface_type='Dirt',
            distance_txt='5½ Furlongs'
        )
        
        print("\n" + "="*80)
        print("PREDICTIONS TABLE:")
        print("="*80)
        print(predictions[['Predicted_Finish', 'Horse', 'Post', 'Rating', 'Probability', 'Fair_Odds']].to_string(index=False))
        
        # Print summary
        print(bridge.generate_prediction_summary(predictions))
        
        # Print parsing report
        report = bridge.get_parse_report()
        print(f"\n{'='*80}")
        print(f"PARSING QUALITY REPORT:")
        print(f"{'='*80}")
        print(f"Overall Confidence: {report['overall_confidence']:.1%}")
        print(f"Horses Parsed: {report['horses_parsed']}")
        if report['issues']:
            print(f"\nIssues ({len(report['issues'])}):")
            for issue in report['issues'][:5]:
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
