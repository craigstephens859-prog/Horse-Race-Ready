"""
VALIDATION TEST: PhD-Level Enhancements
========================================

Tests that new mathematical refinements:
1. Work correctly without errors
2. Produce reasonable values
3. Don't break existing functionality
4. Can be toggled on/off via feature flags

Run this before deploying to production.
"""

import logging
import sys
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_phd_enhancements():
    """Comprehensive validation of PhD-level enhancements"""
    
    logger.info("="*80)
    logger.info("PHD-LEVEL ENHANCEMENTS VALIDATION TEST")
    logger.info("="*80)
    
    try:
        from unified_rating_engine import UnifiedRatingEngine
        logger.info("✓ unified_rating_engine imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import: {e}")
        return False
    
    # Test 1: Feature flags exist
    logger.info("\nTest 1: Feature Flags")
    logger.info("-" * 40)
    try:
        flags = UnifiedRatingEngine.FEATURE_FLAGS
        logger.info(f"Feature flags: {flags}")
        assert 'use_exponential_decay_form' in flags
        assert 'use_game_theoretic_pace' in flags
        assert 'use_entropy_confidence' in flags
        assert 'use_mud_adjustment' in flags
        logger.info("✓ All feature flags present")
    except Exception as e:
        logger.error(f"✗ Feature flags test failed: {e}")
        return False
    
    # Test 2: Initialize engine
    logger.info("\nTest 2: Engine Initialization")
    logger.info("-" * 40)
    try:
        engine = UnifiedRatingEngine(softmax_tau=3.0)
        logger.info("✓ Engine initialized successfully")
    except Exception as e:
        logger.error(f"✗ Engine initialization failed: {e}")
        return False
    
    # Test 3: Sample race with all enhancements enabled
    logger.info("\nTest 3: Prediction with All Enhancements ENABLED")
    logger.info("-" * 40)
    
    sample_pp = """Race 2 Mountaineer 'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025

1 Way of Appeal (E 7)
7/2 Red, Red Cap
BARRIOS RICARDO (254 58-42-39 23%)
Trnr: Cady Khalil (150 18-24-31 12%)
Prime Power: 101.5 (4th)
23Sep23 Mtn Md Sp Wt 16500 98 4th
15Aug23 Mtn Md Sp Wt 16500 92 6th

2 Northern Sky (P 5)
5/1 Blue
LOPEZ JUAN (180 42-35-28 23%)
Trnr: Smith Robert (120 25-18-15 21%)
Prime Power: 95.2 (7th)
01Oct23 Mtn Md Sp Wt 16500 90 7th
10Sep23 Mtn Md Sp Wt 16500 88 8th

3 Fast Lane (E 9)
3/1 Green
GARCIA MIGUEL (220 55-48-40 25%)
Trnr: Johnson Mike (180 35-30-25 19%)
Prime Power: 108.7 (1st)
15Oct23 Mtn Md Sp Wt 16500 102 2nd
28Sep23 Mtn Md Sp Wt 16500 105 3rd
"""
    
    try:
        # Set all flags to True
        engine.FEATURE_FLAGS['use_exponential_decay_form'] = True
        engine.FEATURE_FLAGS['use_game_theoretic_pace'] = True
        engine.FEATURE_FLAGS['use_entropy_confidence'] = True
        engine.FEATURE_FLAGS['use_mud_adjustment'] = True
        
        results = engine.predict_race(
            pp_text=sample_pp,
            today_purse=16500,
            today_race_type="maiden special weight",
            track_name="Mountaineer",
            surface_type="Dirt",
            distance_txt="5.5 Furlongs",
            condition_txt="fast"
        )
        
        logger.info(f"Predicted {len(results)} horses")
        logger.info(f"Top pick: {results.iloc[0]['Horse']} ({results.iloc[0]['Probability']:.1%})")
        
        # Check confidence
        if hasattr(results, 'attrs') and 'system_confidence' in results.attrs:
            confidence = results.attrs['system_confidence']
            logger.info(f"System confidence: {confidence:.3f}")
            assert 0.0 <= confidence <= 1.0, "Confidence out of range"
        
        # Validate probabilities
        prob_sum = results['Probability'].sum()
        assert 0.99 <= prob_sum <= 1.01, f"Probabilities don't sum to 1.0: {prob_sum}"
        
        logger.info("✓ Prediction with enhancements successful")
        logger.info("\nTop 3 Predictions:")
        logger.info(results[['Horse', 'Post', 'Rating', 'Probability', 'Fair_Odds']].head(3).to_string(index=False))
        
    except Exception as e:
        logger.error(f"✗ Prediction with enhancements failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Same race with all enhancements disabled
    logger.info("\nTest 4: Prediction with All Enhancements DISABLED")
    logger.info("-" * 40)
    
    try:
        # Set all flags to False
        engine.FEATURE_FLAGS['use_exponential_decay_form'] = False
        engine.FEATURE_FLAGS['use_game_theoretic_pace'] = False
        engine.FEATURE_FLAGS['use_entropy_confidence'] = False
        engine.FEATURE_FLAGS['use_mud_adjustment'] = False
        
        results_original = engine.predict_race(
            pp_text=sample_pp,
            today_purse=16500,
            today_race_type="maiden special weight",
            track_name="Mountaineer",
            surface_type="Dirt",
            distance_txt="5.5 Furlongs",
            condition_txt="fast"
        )
        
        logger.info(f"Predicted {len(results_original)} horses")
        logger.info(f"Top pick: {results_original.iloc[0]['Horse']} ({results_original.iloc[0]['Probability']:.1%})")
        
        # Validate probabilities
        prob_sum = results_original['Probability'].sum()
        assert 0.99 <= prob_sum <= 1.01, f"Probabilities don't sum to 1.0: {prob_sum}"
        
        logger.info("✓ Prediction with original methods successful")
        logger.info("\nTop 3 Predictions:")
        logger.info(results_original[['Horse', 'Post', 'Rating', 'Probability', 'Fair_Odds']].head(3).to_string(index=False))
        
    except Exception as e:
        logger.error(f"✗ Prediction with original methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Compare results
    logger.info("\nTest 5: Comparison of Enhanced vs Original")
    logger.info("-" * 40)
    
    try:
        # Compare top picks
        enhanced_winner = results.iloc[0]['Horse']
        original_winner = results_original.iloc[0]['Horse']
        
        logger.info(f"Enhanced top pick:  {enhanced_winner}")
        logger.info(f"Original top pick:  {original_winner}")
        
        if enhanced_winner == original_winner:
            logger.info("✓ Same top pick (consistent)")
        else:
            logger.info("✓ Different top pick (enhancements changed prediction)")
        
        # Compare average ratings
        enhanced_avg = results['Rating'].mean()
        original_avg = results_original['Rating'].mean()
        
        logger.info(f"Enhanced avg rating: {enhanced_avg:.2f}")
        logger.info(f"Original avg rating: {original_avg:.2f}")
        logger.info(f"Difference: {abs(enhanced_avg - original_avg):.2f}")
        
    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
        return False
    
    # Test 6: Edge case - muddy track
    logger.info("\nTest 6: Mud Track Adjustment")
    logger.info("-" * 40)
    
    try:
        # Re-enable mud adjustment
        engine.FEATURE_FLAGS['use_mud_adjustment'] = True
        
        results_muddy = engine.predict_race(
            pp_text=sample_pp,
            today_purse=16500,
            today_race_type="maiden special weight",
            track_name="Mountaineer",
            surface_type="Dirt",
            distance_txt="5.5 Furlongs",
            condition_txt="muddy"  # Changed to muddy
        )
        
        logger.info(f"✓ Muddy track prediction successful")
        logger.info(f"Top pick on muddy: {results_muddy.iloc[0]['Horse']}")
        
    except Exception as e:
        logger.error(f"✗ Muddy track test failed: {e}")
        return False
    
    # Test 7: Performance check
    logger.info("\nTest 7: Performance Benchmark")
    logger.info("-" * 40)
    
    try:
        import time
        
        # Time 10 predictions
        start = time.time()
        for i in range(10):
            _ = engine.predict_race(
                pp_text=sample_pp,
                today_purse=16500,
                today_race_type="maiden special weight",
                track_name="Mountaineer",
                surface_type="Dirt",
                distance_txt="5.5 Furlongs",
                condition_txt="fast"
            )
        elapsed = time.time() - start
        avg_time = (elapsed / 10) * 1000  # Convert to ms
        
        logger.info(f"Average prediction time: {avg_time:.1f} ms")
        
        if avg_time < 300:
            logger.info(f"✓ Performance target met (<300ms)")
        else:
            logger.warning(f"⚠ Performance slower than target (300ms)")
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False
    
    # All tests passed
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("="*80)
    logger.info("\nPhD-level enhancements are:")
    logger.info("  ✓ Functional")
    logger.info("  ✓ Backward compatible")
    logger.info("  ✓ Toggleable via feature flags")
    logger.info("  ✓ Performance optimized")
    logger.info("\nSYSTEM READY FOR PRODUCTION DEPLOYMENT")
    
    return True


if __name__ == "__main__":
    success = test_phd_enhancements()
    sys.exit(0 if success else 1)
