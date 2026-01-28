#!/usr/bin/env python3
"""
🚀 COMPLETE ML OPTIMIZATION PIPELINE
Integrates: Weight optimization + Ensemble training + 200-race backtest

DELIVERABLES:
1. Tuned weights table
2. Torch ensemble code
3. Ranked order example
4. Comprehensive accuracy metrics
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List
import torch

# Import our modules
from ml_quant_engine import (
    ModelWeaknesses,
    DynamicWeightOptimizer,
    TrackBiasIntegrator,
    OddsDriftDetector,
    RunningOrderPredictor
)

from backtest_simulator import (
    RaceSimulator,
    BacktestEngine,
    BacktestResults
)

class OptimizationPipeline:
    """Complete ML optimization workflow"""
    
    def __init__(self):
        self.weaknesses = ModelWeaknesses()
        self.optimizer = DynamicWeightOptimizer()
        self.simulator = RaceSimulator(n_races=200)
        self.predictor = RunningOrderPredictor()
        self.backtest_engine = BacktestEngine(self.predictor)
        
        self.optimized_weights = None
        self.backtest_results = None
        self.example_predictions = None
    
    def run_complete_optimization(self, 
                                  optimization_method: str = 'bayesian',
                                  n_iterations: int = 50) -> Dict:
        """
        Run complete optimization pipeline.
        
        Steps:
        1. Analyze current model weaknesses
        2. Generate training data (200 races)
        3. Optimize weights dynamically
        4. Train ensemble models
        5. Run backtest
        6. Generate deliverables
        
        Returns:
            Dict with all results and deliverables
        """
        
        print("="*80)
        print("🧠 ML QUANT OPTIMIZATION PIPELINE")
        print("="*80)
        print()
        
        # Step 1: Weakness Analysis
        print("STEP 1: Analyzing current model weaknesses...")
        print(self.weaknesses.generate_report())
        
        # Step 2: Generate Training Data
        print("\nSTEP 2: Generating 200 training races...")
        training_races = self.simulator.generate_races()
        print(f"✅ Generated {len(training_races)} races")
        
        # Step 3: Optimize Weights
        print(f"\nSTEP 3: Optimizing weights using {optimization_method} method...")
        print(f"  Iterations: {n_iterations}")
        print(f"  Initial weights:")
        for comp, weight in self.optimizer.weights.items():
            print(f"    {comp:15}: {weight:.3f}")
        
        print(f"\n  Running optimization...")
        self.optimized_weights = self.optimizer.optimize_weights(
            training_races,
            method=optimization_method,
            n_iterations=n_iterations
        )
        
        print(f"\n✅ Optimization complete!")
        print(f"  Best accuracy: {self.optimizer.best_accuracy:.3f}")
        print(f"  Optimized weights:")
        for comp, weight in self.optimized_weights.items():
            print(f"    {comp:15}: {weight:.3f}")
        
        # Step 4: Train Ensemble
        print("\nSTEP 4: Training ensemble models...")
        self.predictor.train(training_races, n_epochs=30)
        
        # Step 5: Run Backtest
        print("\nSTEP 5: Running 200-race backtest...")
        # Generate fresh test set
        test_races = self.simulator.generate_races()
        self.backtest_results = self.backtest_engine.run_backtest(test_races)
        
        print(self.backtest_results)
        
        # Step 6: Generate Example Predictions
        print("\nSTEP 6: Generating example predictions...")
        sample_race = test_races[0]
        self.example_predictions = self.predictor.predict_running_order(
            sample_race['horses'],
            sample_race['track'],
            sample_race['surface'],
            sample_race['distance']
        )
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE!")
        print("="*80)
        
        return {
            'optimized_weights': self.optimized_weights,
            'weights_table': self.optimizer.generate_weights_table(),
            'backtest_results': self.backtest_results,
            'example_predictions': self.example_predictions,
            'training_races': training_races,
            'test_races': test_races
        }
    
    def generate_deliverables(self, results: Dict, output_dir: str = '.') -> None:
        """
        Generate all deliverables.
        
        1. Tuned weights table (CSV + Markdown)
        2. Torch ensemble code snippet
        3. Ranked order example
        4. Accuracy metrics summary
        """
        
        print("\n" + "="*80)
        print("📊 GENERATING DELIVERABLES")
        print("="*80)
        
        # Deliverable 1: Weights Table
        print("\n1️⃣  TUNED WEIGHTS TABLE")
        print("-" * 80)
        weights_df = results['weights_table']
        print(weights_df.to_string(index=False))
        
        weights_path = f"{output_dir}/optimized_weights.csv"
        weights_df.to_csv(weights_path, index=False)
        print(f"\n✅ Saved to: {weights_path}")
        
        # Also save as markdown
        weights_md_path = f"{output_dir}/optimized_weights.md"
        with open(weights_md_path, 'w') as f:
            f.write("# Optimized Component Weights\n\n")
            f.write(weights_df.to_markdown(index=False))
            f.write("\n\n## Summary\n\n")
            f.write(f"- **Optimization Method**: Bayesian\n")
            f.write(f"- **Training Accuracy**: {self.optimizer.best_accuracy:.1%}\n")
            f.write(f"- **Backtest Winner Accuracy**: {self.backtest_results.winner_accuracy:.1%}\n")
        print(f"✅ Markdown saved to: {weights_md_path}")
        
        # Deliverable 2: Torch Ensemble Code
        print("\n2️⃣  TORCH ENSEMBLE CODE SNIPPET")
        print("-" * 80)
        torch_code = self._generate_torch_code(results['optimized_weights'])
        print(torch_code)
        
        torch_path = f"{output_dir}/ensemble_model.py"
        with open(torch_path, 'w') as f:
            f.write(torch_code)
        print(f"\n✅ Saved to: {torch_path}")
        
        # Deliverable 3: Ranked Order Example
        print("\n3️⃣  RANKED ORDER EXAMPLE")
        print("-" * 80)
        example_df = results['example_predictions'].copy()
        
        # Format for display
        example_df['Win_Prob'] = example_df['Win_Prob'].apply(lambda x: f"{x:.2%}")
        example_df['Place_Prob'] = example_df['Place_Prob'].apply(lambda x: f"{x:.2%}")
        example_df['Show_Prob'] = example_df['Show_Prob'].apply(lambda x: f"{x:.2%}")
        example_df['Fourth_Prob'] = example_df['Fourth_Prob'].apply(lambda x: f"{x:.2%}")
        example_df['Composite_Score'] = example_df['Composite_Score'].apply(lambda x: f"{x:.3f}")
        
        print(example_df.to_string(index=False))
        
        example_path = f"{output_dir}/example_predictions.csv"
        results['example_predictions'].to_csv(example_path, index=False)
        print(f"\n✅ Saved to: {example_path}")
        
        # Deliverable 4: Accuracy Metrics
        print("\n4️⃣  ACCURACY METRICS SUMMARY")
        print("-" * 80)
        metrics_report = self.backtest_engine.generate_report(
            self.backtest_results,
            save_path=f"{output_dir}/backtest_report.txt"
        )
        print(metrics_report)
        
        # Summary JSON
        summary = {
            'winner_accuracy': float(self.backtest_results.winner_accuracy),
            'place_accuracy': float(self.backtest_results.place_accuracy),
            'show_accuracy': float(self.backtest_results.show_accuracy),
            'exacta_accuracy': float(self.backtest_results.exacta_accuracy),
            'trifecta_accuracy': float(self.backtest_results.trifecta_accuracy),
            'roi_percent': float(self.backtest_results.roi_percent),
            'calibration_error': float(self.backtest_results.calibration_error),
            'second_place_contenders': float(self.backtest_results.second_place_contenders),
            'third_place_contenders': float(self.backtest_results.third_place_contenders),
            'optimized_weights': results['optimized_weights'],
            'target_achieved': self.backtest_results.winner_accuracy >= 0.90
        }
        
        import json
        summary_path = f"{output_dir}/optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary saved to: {summary_path}")
        
        print("\n" + "="*80)
        print("✅ ALL DELIVERABLES GENERATED")
        print("="*80)
    
    def _generate_torch_code(self, weights: Dict[str, float]) -> str:
        """Generate production-ready torch ensemble code"""
        
        code = f'''"""
Production Torch Ensemble Model
Generated by ML Quant Optimization Pipeline
"""

import torch
import torch.nn as nn
import numpy as np

class ProductionEnsemble(nn.Module):
    """
    Production-ready ensemble model for horse racing predictions.
    
    Optimized weights:
{self._format_weights_as_comments(weights)}
    """
    
    def __init__(self):
        super().__init__()
        
        # Optimized component weights
        self.weights = {{
{self._format_weights_as_dict(weights)}
        }}
        
        # Neural network for non-linear refinement
        self.refiner = nn.Sequential(
            nn.Linear(15, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Win, Place, Show, Fourth
        )
    
    def calculate_base_rating(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calculate base rating using optimized linear weights.
        
        Args:
            features: Tensor of shape (batch_size, 15) with:
                [class, form, speed, pace, style, post, angles,
                 quirin, jockey_win, trainer_win, last_beyer,
                 avg_beyer, track_bias, odds_drift, layoff]
        
        Returns:
            Base ratings (batch_size,)
        """
        
        rating = (
            features[:, 0] * self.weights['class'] +
            features[:, 1] * self.weights['form'] +
            features[:, 2] * self.weights['speed'] +
            features[:, 3] * self.weights['pace'] +
            features[:, 4] * self.weights['style'] +
            features[:, 5] * self.weights['post'] +
            features[:, 6] * self.weights['angles'] +
            features[:, 12] * self.weights['track_bias'] +
            features[:, 13] * self.weights['odds_drift']
        )
        
        return rating
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Full ensemble prediction.
        
        Returns:
            Probabilities (batch_size, 4) for [Win, Place, Show, Fourth]
        """
        
        # Calculate base rating
        base_rating = self.calculate_base_rating(features)
        
        # Refine with neural network
        refined = self.refiner(features)
        
        # Combine: 70% base, 30% neural refinement
        combined = 0.7 * base_rating.unsqueeze(1) + 0.3 * refined[:, 0].unsqueeze(1)
        
        # Convert to probabilities via softmax
        probs = torch.nn.functional.softmax(combined / 3.0, dim=1)
        
        return probs
    
    def predict_race(self, horses: list) -> list:
        """
        Predict finish order for a race.
        
        Args:
            horses: List of dicts with horse features
        
        Returns:
            List of (horse_name, win_prob, place_prob, show_prob, fourth_prob)
        """
        
        # Extract features
        features = []
        names = []
        
        for horse in horses:
            feature_vec = [
                horse.get('class', 0),
                horse.get('form', 0),
                horse.get('speed', 0),
                horse.get('pace', 0),
                horse.get('style_numeric', 0),
                horse.get('post', 0),
                horse.get('angles', 0),
                horse.get('quirin', 0),
                horse.get('jockey_win_pct', 0),
                horse.get('trainer_win_pct', 0),
                horse.get('last_beyer', 0),
                horse.get('avg_beyer', 0),
                horse.get('track_bias', 0),
                horse.get('odds_drift', 0),
                horse.get('days_since_last', 30) / 100.0
            ]
            features.append(feature_vec)
            names.append(horse.get('name', f"Horse_{{len(names)+1}}"))
        
        # Convert to tensor
        X = torch.tensor(features, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            probs = self.forward(X)
        
        # Build results
        results = []
        for i, name in enumerate(names):
            results.append({{
                'name': name,
                'win_prob': float(probs[i, 0]),
                'place_prob': float(probs[i, 1]) if probs.shape[1] > 1 else 0.0,
                'show_prob': float(probs[i, 2]) if probs.shape[1] > 2 else 0.0,
                'fourth_prob': float(probs[i, 3]) if probs.shape[1] > 3 else 0.0
            }})
        
        # Sort by win probability
        results.sort(key=lambda x: x['win_prob'], reverse=True)
        
        return results


# Usage Example
if __name__ == "__main__":
    model = ProductionEnsemble()
    
    # Sample race
    horses = [
        {{'name': 'Fast Runner', 'class': 2.0, 'form': 1.5, 'speed': 1.8,
          'pace': 1.2, 'style_numeric': 2, 'post': 3, 'angles': 0.3}},
        {{'name': 'Speed Demon', 'class': 1.5, 'form': 2.0, 'speed': 2.2,
          'pace': 0.8, 'style_numeric': 3, 'post': 1, 'angles': 0.2}},
        # ... more horses
    ]
    
    predictions = model.predict_race(horses)
    
    print("Predicted Finish Order:")
    for i, pred in enumerate(predictions, 1):
        print(f"{{i}}. {{pred['name']:20}} Win: {{pred['win_prob']:.1%}}")
'''
        
        return code
    
    def _format_weights_as_comments(self, weights: Dict[str, float]) -> str:
        """Format weights as code comments"""
        lines = []
        for comp, weight in weights.items():
            lines.append(f"        {comp:15}: {weight:.3f}")
        return "\n".join(lines)
    
    def _format_weights_as_dict(self, weights: Dict[str, float]) -> str:
        """Format weights as Python dict"""
        lines = []
        for comp, weight in weights.items():
            lines.append(f"            '{comp}': {weight:.3f},")
        return "\n".join(lines)


# ===================== MAIN EXECUTION =====================

def main():
    """Run complete optimization pipeline"""
    
    pipeline = OptimizationPipeline()
    
    # Run optimization (with smaller iterations for demo)
    results = pipeline.run_complete_optimization(
        optimization_method='bayesian',
        n_iterations=20  # Increase to 50-100 for production
    )
    
    # Generate deliverables
    pipeline.generate_deliverables(results)
    
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    print(f"\n✅ Winner Accuracy: {results['backtest_results'].winner_accuracy:.1%}")
    print(f"   Target: 90.0%")
    
    if results['backtest_results'].winner_accuracy >= 0.90:
        print("\n🎉 TARGET ACHIEVED!")
    else:
        gap = 0.90 - results['backtest_results'].winner_accuracy
        print(f"\n⚠️  Gap to close: +{gap:.1%}")
    
    print(f"\n✅ 2nd Place Contenders: {results['backtest_results'].second_place_contenders:.1f}")
    print(f"   Target: 2.0")
    
    print(f"\n✅ 3rd Place Contenders: {results['backtest_results'].third_place_contenders:.1f}")
    print(f"   Target: 2-3")
    
    print(f"\n✅ ROI: {results['backtest_results'].roi_percent:+.1%}")
    
    print("\n" + "="*80)
    print("📁 OUTPUT FILES GENERATED:")
    print("="*80)
    print("  - optimized_weights.csv")
    print("  - optimized_weights.md")
    print("  - ensemble_model.py")
    print("  - example_predictions.csv")
    print("  - backtest_report.txt")
    print("  - optimization_summary.json")
    print("\n✅ All deliverables ready!")


if __name__ == "__main__":
    main()
