# INTEGRATION INSTRUCTIONS: app.py → Unified Rating Engine
# Drop-in replacement for Section B rating calculation

## STEP 1: Add import at top of app.py (around line 50)

```python
# ULTRATHINK INTEGRATION: Unified rating engine
try:
    from unified_rating_engine import UnifiedRatingEngine
    UNIFIED_ENGINE_AVAILABLE = True
except ImportError:
    UNIFIED_ENGINE_AVAILABLE = False
    UnifiedRatingEngine = None
```

## STEP 2: Replace Section B rating calculation (around line 2276)

### FIND THIS CODE:
```python
ratings_df = compute_bias_ratings(
    df_final_field,
    surface_type=surface_type,
    distance_txt=distance_txt,
    condition_txt=condition_txt,
    race_type=race_type_detected,
    running_style_bias=running_style_biases[0] if running_style_biases else "fair/neutral",
    post_bias_pick=post_biases[0] if post_biases else "no significant post bias",
    pedigree_per_horse=pedigree_per_horse,
    track_name=track_name,
    pp_text=pp_text,
    figs_df=figs_df
)
```

### REPLACE WITH:
```python
# ULTRATHINK: Use unified engine if available, fallback to legacy system
if UNIFIED_ENGINE_AVAILABLE:
    try:
        st.info("🎯 Using Unified Rating Engine (Gold Standard)")
        
        # Initialize engine
        engine = UnifiedRatingEngine(softmax_tau=3.0)
        
        # Generate predictions
        ratings_df = engine.predict_race(
            pp_text=pp_text,
            today_purse=purse_val,
            today_race_type=race_type_detected,
            track_name=track_name,
            surface_type=surface_type,
            distance_txt=distance_txt,
            condition_txt=condition_txt,
            style_bias=running_style_biases if running_style_biases else None,
            post_bias=post_biases if post_biases else None
        )
        
        # Display parsing confidence
        if engine.last_validation:
            conf = engine.last_validation['overall_confidence']
            if conf >= 0.9:
                st.success(f"✅ Parsing Confidence: {conf:.1%} (Excellent)")
            elif conf >= 0.7:
                st.info(f"ℹ️ Parsing Confidence: {conf:.1%} (Good)")
            else:
                st.warning(f"⚠️ Parsing Confidence: {conf:.1%} (Review recommended)")
                
                # Show issues
                if engine.last_validation['critical_issues']:
                    with st.expander("View Parsing Issues"):
                        for issue in engine.last_validation['critical_issues']:
                            st.write(f"- {issue}")
        
        # Rename columns to match legacy format
        ratings_df = ratings_df.rename(columns={
            'Probability': 'FairProb',
            'Predicted_Finish': 'Rank'
        })
        
        # Add '#' column if missing
        if '#' not in ratings_df.columns:
            ratings_df['#'] = ratings_df['Post']
            
    except Exception as e:
        st.error(f"❌ Unified engine failed: {e}")
        st.info("Falling back to legacy rating system...")
        
        # Fallback to legacy system
        ratings_df = compute_bias_ratings(
            df_final_field,
            surface_type=surface_type,
            distance_txt=distance_txt,
            condition_txt=condition_txt,
            race_type=race_type_detected,
            running_style_bias=running_style_biases[0] if running_style_biases else "fair/neutral",
            post_bias_pick=post_biases[0] if post_biases else "no significant post bias",
            pedigree_per_horse=pedigree_per_horse,
            track_name=track_name,
            pp_text=pp_text,
            figs_df=figs_df
        )
else:
    # No unified engine available, use legacy
    st.info("Using Legacy Rating System")
    ratings_df = compute_bias_ratings(
        df_final_field,
        surface_type=surface_type,
        distance_txt=distance_txt,
        condition_txt=condition_txt,
        race_type=race_type_detected,
        running_style_bias=running_style_biases[0] if running_style_biases else "fair/neutral",
        post_bias_pick=post_biases[0] if post_biases else "no significant post bias",
        pedigree_per_horse=pedigree_per_horse,
        track_name=track_name,
        pp_text=pp_text,
        figs_df=figs_df
    )
```

## STEP 3: Update display section to show enhanced metrics

### FIND THIS CODE (around line 2300):
```python
st.dataframe(ratings_df, use_container_width=True)
```

### ADD BEFORE IT:
```python
# Show enhanced metrics if using unified engine
if UNIFIED_ENGINE_AVAILABLE and 'Angles_Total' in ratings_df.columns:
    st.subheader("📊 Rating Component Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Parsing Confidence", 
                 f"{ratings_df['Parse_Confidence'].mean():.1%}")
    
    with col2:
        st.metric("Average Angle Score", 
                 f"{ratings_df['Angles_Total'].mean():.2f}/8.0")
    
    with col3:
        top_horse = ratings_df.iloc[0]
        st.metric("Top Selection Win Probability",
                 f"{top_horse['FairProb']:.1%}")
    
    # Component breakdown table for top 3
    with st.expander("View Top 3 Rating Components"):
        top3 = ratings_df.head(3)
        component_cols = ['Horse', 'Cclass', 'Cform', 'Cspeed', 'Cpace', 'Cstyle', 'Cpost', 'Angles_Total', 'Tier2_Bonus', 'Rating']
        available_cols = [col for col in component_cols if col in top3.columns]
        st.dataframe(top3[available_cols], use_container_width=True)

# Original dataframe display
st.dataframe(ratings_df, use_container_width=True)
```

## STEP 4: Test the integration

### Run Streamlit:
```bash
streamlit run app.py
```

### Verify:
1. ✅ "Using Unified Rating Engine" message appears
2. ✅ Parsing confidence displayed
3. ✅ Component breakdown shows for top 3
4. ✅ Rating, Probability, Fair_Odds columns present
5. ✅ No errors in console

### If errors occur:
1. Check import statement (line 50)
2. Verify unified_rating_engine.py is in same directory
3. Check that horse_angles8.py has been replaced with optimized version
4. Review console for specific error messages

## STEP 5: Compare results

### Legacy System Output:
```
Horse          R    FairProb  Fair_Odds
Horse A       15.2   0.48      2.08
Horse B       13.1   0.26      3.85
Horse C       11.8   0.17      5.88
```

### Unified Engine Output:
```
Horse          Rating  FairProb  Fair_Odds  Confidence
Horse A        16.08    0.598     1.67       0.95
Horse B        12.30    0.169     5.92       0.92
Horse C        11.10    0.112     8.93       0.88
```

### Key Differences:
- **Ratings**: Unified engine uses weighted formula (class×2.5, form×1.8, etc.)
- **Probabilities**: Softmax applied with tau=3.0
- **Confidence**: Shows parsing quality per horse
- **Components**: Full breakdown available (Cclass, Cform, Cspeed, etc.)

## ADVANCED: Custom Configuration

### Adjust softmax temperature:
```python
# More concentrated (favorites favored)
engine = UnifiedRatingEngine(softmax_tau=2.0)

# More uniform (even field)
engine = UnifiedRatingEngine(softmax_tau=4.0)

# Balanced (recommended)
engine = UnifiedRatingEngine(softmax_tau=3.0)
```

### Adjust component weights:
```python
# In unified_rating_engine.py, modify WEIGHTS dict:
WEIGHTS = {
    'class': 3.0,  # Increase class importance
    'speed': 2.0,
    'form': 1.5,   # Decrease form importance
    'pace': 1.5,
    'style': 1.2,
    'post': 0.8,
    'angles': 0.15  # Increase angle influence
}
```

### Enable debug mode:
```python
# See step-by-step calculations
angles_df = compute_eight_angles(df, use_weights=True, debug=True)
```

## TROUBLESHOOTING

### Issue: "No horses could be parsed"
**Solution:** Check PP text format, ensure BRISNET standard format

### Issue: "ModuleNotFoundError: No module named 'unified_rating_engine'"
**Solution:** Ensure unified_rating_engine.py is in same directory as app.py

### Issue: Parsing confidence < 0.7
**Solution:** 
1. Review parsing issues in expander
2. Check for typos in PP text
3. Ensure all horses have jockey/trainer data
4. Manually correct ML odds if missing

### Issue: NaN in rating components
**Solution:** Check that horse_angles8.py has been replaced with optimized version

### Issue: Probabilities don't sum to 1.0
**Solution:** Softmax automatically normalizes, but check for scratched horses in data

## VALIDATION

### Before deploying to production:
1. ✅ Test with 5+ different races
2. ✅ Verify all horses get ratings
3. ✅ Check parsing confidence > 0.7 average
4. ✅ Ensure no NaN/Inf in output
5. ✅ Compare top selections to intuition
6. ✅ Validate fair odds are reasonable

### Success Criteria:
- Parsing confidence > 0.70 average
- All horses have complete ratings
- Top selection has compelling story (class, form, speed advantages)
- Fair odds align with ML odds (within 2-3 points)
- No console errors during execution

## NEXT STEPS

After successful integration:
1. Capture 10 races via Section F
2. Record actual results
3. Compare predictions vs outcomes
4. Calculate accuracy metrics
5. Adjust weights if needed
6. Continue accumulation toward 90% accuracy goal

---

**Integration Time:** 15-30 minutes  
**Testing Time:** 10-15 minutes per race  
**Production Ready:** After 5+ successful test races  

**Questions? Check:**
- ULTRATHINK_CONSOLIDATION_COMPLETE.md
- INTEGRATION_GUIDE.md  
- unified_rating_engine.py (inline comments)
