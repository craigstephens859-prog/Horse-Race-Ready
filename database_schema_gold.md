# Gold-Standard Historical Racing Database Schema
## Target: 90%+ Top-5 Prediction Accuracy

**Design Philosophy**: Zero omissions. Every parameter from Equibase, TrackMaster, and BRISNET that impacts top-5 finishing order.

---

## Core Tables

### 1. RACES (Race-Level Metadata)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| race_id | TEXT PK | Generated | Y | {track}_{date}_{race_num} unique identifier |
| track_code | TEXT | Equibase/BRIS | Y | 3-letter code (GP, SA, KEE) |
| race_date | DATE | Equibase/BRIS | Y | YYYY-MM-DD |
| race_number | INT | Equibase/BRIS | Y | 1-12 typically |
| post_time | TIME | Equibase | N | Actual post time |
| distance_furlongs | FLOAT | Equibase/BRIS | Y | 5.0, 6.0, 8.5, 10.0, etc. |
| distance_yards | INT | Equibase | Y | 880, 1320, etc. (precise) |
| surface | TEXT | Equibase/BRIS | Y | D (dirt), T (turf), AW (all-weather) |
| track_condition | TEXT | Equibase/BRIS | Y | FT, GD, YL, SY, FM, MY, etc. |
| rail_distance | INT | Equibase | Y | Feet from inner rail (turf courses) |
| run_up_distance | INT | Equibase | Y | Feet from gate to timing start |
| temp_high | INT | Equibase | N | Fahrenheit |
| weather | TEXT | Equibase | N | Clear, cloudy, rain, etc. |
| purse | INT | Equibase/BRIS | Y | USD purse value |
| race_type | TEXT | Equibase/BRIS | Y | MCL, CLM, ALW, STK, G1, G2, G3, etc. |
| race_class_level | INT | Derived | Y | 1-8 normalized class score |
| claiming_price | INT | Equibase/BRIS | Y | If CLM/MCL race |
| sex_restriction | TEXT | Equibase | Y | M (males), F (fillies/mares), blank=open |
| age_restriction | TEXT | Equibase | Y | 2YO, 3YO, 3+, etc. |
| breed | TEXT | Equibase | N | TB (thoroughbred), QH, AR |
| field_size | INT | Equibase/BRIS | Y | Number of starters |
| race_shape | TEXT | Derived | Y | E-heavy, P-heavy, balanced, contested |
| fractional_1 | FLOAT | Equibase | Y | First call time (e.g., 22.14 for 2f) |
| fractional_2 | FLOAT | Equibase | Y | Second call time |
| fractional_3 | FLOAT | Equibase | Y | Third call (if applicable) |
| final_time | FLOAT | Equibase/BRIS | Y | Official final time |
| track_variant | INT | Equibase | Y | Daily speed variant |
| equibase_speed_figure | INT | Equibase | Y | ESF for winner/race |
| bris_track_variant | INT | BRISNET | Y | BRIS track variant |
| comments | TEXT | Equibase | N | Race narrative/notes |
| video_link | TEXT | Equibase | N | Video replay URL |
| scratch_count | INT | Equibase | Y | Number scratched |
| winner_margin | FLOAT | Equibase | Y | Winning margin (lengths) |
| winner_odds | FLOAT | Equibase | Y | Final odds of winner |
| show_payoff | FLOAT | Equibase | N | $2 show payout for winner |
| pace_scenario | TEXT | Derived | Y | FAST, CONTESTED, HONEST, CRAWL |
| track_bias_flags | TEXT | Derived | Y | JSON: {"speed_favored": true, "rail_dead": false} |

---

### 2. RUNNERS (Horse-Race Instance)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| runner_id | TEXT PK | Generated | Y | {race_id}_{program_num} |
| race_id | TEXT FK | - | Y | Links to RACES |
| horse_id | TEXT FK | - | Y | Links to HORSES master table |
| program_number | INT | Equibase/BRIS | Y | Post position/program # (1-20) |
| post_position | INT | Equibase/BRIS | Y | Actual gate position |
| morning_line_odds | FLOAT | Equibase/BRIS | Y | ML odds (e.g., 5.0 = 5-1) |
| final_odds | FLOAT | Equibase | Y | Official SP odds |
| weight_carried | INT | Equibase/BRIS | Y | Pounds (118, 121, 126, etc.) |
| weight_allowance | INT | Equibase | Y | Apprentice/sex allowance |
| jockey_name | TEXT | Equibase/BRIS | Y | Full name |
| jockey_id | TEXT FK | - | Y | Links to JOCKEYS |
| trainer_name | TEXT | Equibase/BRIS | Y | Full name |
| trainer_id | TEXT FK | - | Y | Links to TRAINERS |
| owner_name | TEXT | Equibase | N | Owner |
| medication_lasix | BOOL | Equibase/BRIS | Y | L flag |
| medication_bute | BOOL | Equibase | Y | B flag (historical) |
| equipment_blinkers | TEXT | BRISNET | Y | blank, B, BF, BE (blinkers/first-time/off) |
| equipment_bandages | TEXT | BRISNET | N | Front/hind bandages |
| claimed | BOOL | Equibase | Y | Was claimed this race? |
| claim_price | INT | Equibase | Y | If claimed |
| days_since_last_race | INT | Derived | Y | Layoff days |
| lifetime_starts | INT | Derived | Y | Career starts before this race |
| lifetime_wins | INT | Derived | Y | Career wins |
| lifetime_earnings | INT | Derived | Y | Career $ |
| current_class_rating | FLOAT | Derived | Y | Today's class vs horse avg |
| avg_beyer_last_3 | FLOAT | Derived | Y | Mean of last 3 Beyers |
| avg_beyer_last_5 | FLOAT | Derived | Y | Mean of last 5 Beyers |
| best_beyer_last_12mo | INT | Derived | Y | Peak speed last year |
| running_style | TEXT | Derived/BRIS | Y | E, E/P, P, S (early, presser, etc.) |
| early_pace_rating | INT | BRISNET | Y | E1 rating |
| late_pace_rating | INT | BRISNET | Y | E2/LP ratings |
| bris_speed_rating | INT | BRISNET | Y | BRIS Speed |
| bris_class_rating | INT | BRISNET | Y | BRIS Class |
| prime_power_rating | INT | BRISNET | Y | Prime Power |
| dirt_pedigree_rating | INT | BRISNET | Y | Sire dirt aptitude (0-100) |
| turf_pedigree_rating | INT | BRISNET | Y | Sire turf aptitude |
| mud_pedigree_rating | INT | BRISNET | Y | Mud/off-track pedigree |
| distance_pedigree_rating | INT | BRISNET | Y | Distance (route) pedigree |
| sire_avg_win_dist | FLOAT | BRISNET | Y | Sire progeny avg winning distance |
| dam_produce_record | TEXT | BRISNET | N | Dam's winners/earnings |
| trainer_current_meet_stats | TEXT | Derived | Y | JSON: {"starts": 45, "wins": 9, "roi": 0.82} |
| jockey_current_meet_stats | TEXT | Derived | Y | JSON similar |
| trainer_jockey_combo_roi | FLOAT | Derived | Y | T/J combo ROI if >= 5 starts |
| trainer_surface_win_pct | FLOAT | Derived | Y | Trainer % on this surface |
| jockey_track_win_pct | FLOAT | Derived | Y | Jockey % at this track |
| form_cycle | INT | Derived | Y | Recent form trend: +2 (improving), 0 (stable), -2 (declining) |
| recency_score | FLOAT | Derived | Y | Decay function: max speed × decay factor |
| class_change_delta | FLOAT | Derived | Y | Today's class - avg last 3 (positive = drop) |
| surface_switch_flag | BOOL | Derived | Y | Surface change from last race? |
| distance_switch_flag | BOOL | Derived | Y | +/- 1F+ distance change? |
| jockey_change_flag | BOOL | Derived | Y | Different jockey from last? |
| post_bias_adjustment | FLOAT | Derived | Y | Track/distance-specific post advantage |
| workout_pattern_score | FLOAT | BRISNET/Derived | Y | Recent workout quality (recency, speed) |
| workout_best_recent | TEXT | BRISNET | Y | Best workout last 30 days (e.g., "5f :59.2 B") |

---

### 3. RESULTS (Finish Order - GROUND TRUTH for ML)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| result_id | TEXT PK | Generated | Y | {race_id}_{program_num} |
| race_id | TEXT FK | - | Y | Links to RACES |
| runner_id | TEXT FK | - | Y | Links to RUNNERS |
| program_number | INT | Equibase | Y | Program # |
| finish_position | INT | Equibase | Y | **1-5 = critical labels** |
| official_finish | TEXT | Equibase | Y | "1", "2", "DH" (dead heat), "DQ" |
| disqualified_from | INT | Equibase | Y | If DQ'd, original finish |
| beaten_lengths | FLOAT | Equibase | Y | Lengths behind winner |
| pos_at_start | INT | Equibase | Y | Position at break |
| pos_1st_call | INT | Equibase | Y | Position at 1st call |
| lengths_1st_call | FLOAT | Equibase | Y | Lengths back at 1st call |
| pos_2nd_call | INT | Equibase | Y | Position at 2nd call |
| lengths_2nd_call | FLOAT | Equibase | Y | Lengths back |
| pos_stretch_call | INT | Equibase | Y | Stretch position |
| lengths_stretch | FLOAT | Equibase | Y | Lengths back in stretch |
| pos_finish | INT | Equibase | Y | Final position (redundant check) |
| lengths_finish | FLOAT | Equibase | Y | Final margin |
| equibase_speed_fig_earned | INT | Equibase | Y | ESF for this runner |
| bris_speed_earned | INT | BRISNET | Y | BRIS Speed rating earned |
| final_fraction | FLOAT | Equibase | Y | Horse's final fraction time |
| trip_comment | TEXT | Equibase | Y | "Bumped start, wide turn, rallied" |
| trouble_flags | TEXT | Derived | Y | JSON: {"bumped": true, "checked": true, "wide": false} |
| gain_from_2nd_call | INT | Derived | Y | Positions gained from 2C to finish |
| gain_from_stretch | INT | Derived | Y | Positions from stretch to finish |
| pace_position_score | FLOAT | Derived | Y | How close to ideal pace position? |

---

### 4. PP_LINES (Historical Past Performances - Normalized)
**Each runner has 10-12 PP lines** (one row per past race)

| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| pp_line_id | TEXT PK | Generated | Y | {runner_id}_{pp_index} |
| runner_id | TEXT FK | - | Y | Today's runner |
| pp_index | INT | - | Y | 0=most recent, 11=oldest |
| past_race_date | DATE | BRISNET | Y | Date of this PP |
| past_track_code | TEXT | BRISNET | Y | Track code |
| past_distance | FLOAT | BRISNET | Y | Distance (furlongs) |
| past_surface | TEXT | BRISNET | Y | D/T/AW |
| past_condition | TEXT | BRISNET | Y | FT, SY, etc. |
| past_race_type | TEXT | BRISNET | Y | MCL, ALW, etc. |
| past_class | INT | BRISNET | Y | Class level |
| past_field_size | INT | BRISNET | Y | # starters |
| past_post | INT | BRISNET | Y | Post position |
| past_odds | FLOAT | BRISNET | Y | Final odds |
| past_weight | INT | BRISNET | Y | Weight carried |
| past_jockey | TEXT | BRISNET | Y | Jockey name |
| past_finish_pos | INT | BRISNET | Y | 1-12 or "PU", "F" |
| past_beaten_lengths | FLOAT | BRISNET | Y | Lengths behind winner |
| past_1st_call_pos | INT | BRISNET | Y | Position at 1C |
| past_2nd_call_pos | INT | BRISNET | Y | Position at 2C |
| past_stretch_pos | INT | BRISNET | Y | Stretch position |
| past_final_fraction | FLOAT | BRISNET | Y | Horse's final fraction |
| past_beyer | INT | BRISNET | Y | Beyer Speed Figure |
| past_bris_speed | INT | BRISNET | Y | BRIS Speed |
| past_e1_pace | INT | BRISNET | Y | E1 |
| past_e2_pace | INT | BRISNET | Y | E2 |
| past_late_pace | INT | BRISNET | Y | LP |
| past_class_rating | INT | BRISNET | Y | BRIS Class |
| past_prime_power | INT | BRISNET | Y | Prime Power |
| past_trip_comment | TEXT | BRISNET | Y | "Wide", "Bumped", "Checked" |
| past_medication | TEXT | BRISNET | Y | L (Lasix), B (Bute) |
| past_equipment | TEXT | BRISNET | Y | Blinkers, bandages |
| days_back_from_today | INT | Derived | Y | Days between PP and today's race |

---

### 5. HORSES (Master Horse Registry)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| horse_id | TEXT PK | Equibase/BRIS | Y | Unique ID (Equibase Registry #) |
| horse_name | TEXT | Equibase/BRIS | Y | Registered name |
| foaling_year | INT | BRISNET | Y | Birth year (YOB) |
| sex | TEXT | Equibase/BRIS | Y | C, F, G, H, M |
| color | TEXT | Equibase | N | B (bay), CH (chestnut), etc. |
| sire_name | TEXT | BRISNET | Y | Sire name |
| sire_id | TEXT | - | Y | Sire unique ID |
| dam_name | TEXT | BRISNET | Y | Dam name |
| dam_id | TEXT | - | Y | Dam unique ID |
| sire_sire_name | TEXT | BRISNET | N | Grandsire (sire's sire) |
| breeder | TEXT | Equibase | N | Breeder name |
| lifetime_record | TEXT | Derived | N | "45-12-8-6" (starts-wins-2nd-3rd) |
| lifetime_earnings | INT | Derived | Y | Career $ |
| avg_class_level | FLOAT | Derived | Y | Career average class |
| preferred_surface | TEXT | Derived | Y | D/T based on wins/ROI |
| preferred_distance_range | TEXT | Derived | Y | "Sprint" (< 8.5F), "Route" (>= 8.5F) |

---

### 6. TRACK_BIASES (Track-Specific Adjustments)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| bias_id | TEXT PK | Generated | Y | {track}_{surface}_{distance_bucket}_{date_range} |
| track_code | TEXT | Derived | Y | GP, SA, etc. |
| surface | TEXT | Derived | Y | D, T, AW |
| distance_bucket | TEXT | Derived | Y | "5F-6F", "8F-9F", "10F+" |
| start_date | DATE | Derived | Y | Analysis period start |
| end_date | DATE | Derived | Y | Analysis period end |
| speed_bias | FLOAT | Derived | Y | -1 (closer bias) to +1 (speed bias) |
| rail_bias | FLOAT | Derived | Y | -0.5 (rail dead) to +0.5 (rail golden) |
| post_1_3_win_pct | FLOAT | Derived | Y | Inside posts win % |
| post_8_12_win_pct | FLOAT | Derived | Y | Outside posts win % |
| early_vs_closer_ratio | FLOAT | Derived | Y | E/EP win % vs P/S win % |
| avg_winning_e1 | FLOAT | Derived | Y | Average E1 of winners |
| avg_winning_beyer | FLOAT | Derived | Y | Track speed baseline |

---

### 7. JOCKEYS (Jockey Stats)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| jockey_id | TEXT PK | Generated | Y | Unique ID |
| jockey_name | TEXT | Equibase/BRIS | Y | Full name |
| current_year_starts | INT | Derived | Y | YTD starts |
| current_year_wins | INT | Derived | Y | YTD wins |
| current_year_roi | FLOAT | Derived | Y | $2 ROI |
| track_specific_stats | TEXT | Derived | Y | JSON per track |
| surface_specific_stats | TEXT | Derived | Y | JSON: dirt/turf/AW splits |
| distance_specific_stats | TEXT | Derived | Y | Sprint vs route |

---

### 8. TRAINERS (Trainer Stats)
| Field | Type | Source | Critical? | Description |
|-------|------|--------|-----------|-------------|
| trainer_id | TEXT PK | Generated | Y | Unique ID |
| trainer_name | TEXT | Equibase/BRIS | Y | Full name |
| current_year_starts | INT | Derived | Y | YTD starts |
| current_year_wins | INT | Derived | Y | YTD wins |
| current_year_roi | FLOAT | Derived | Y | $2 ROI |
| layoff_specialty | BOOL | Derived | Y | Win% after 60+ day layoff > 15%? |
| first_time_starter_pct | FLOAT | Derived | Y | FTS win % |
| turf_specialist | BOOL | Derived | Y | Turf win% > 18%? |
| claiming_to_alw_pct | FLOAT | Derived | Y | Success moving horses up |

---

## Engineered Features (30+)

### Speed/Class Features
1. **speed_rating_today** = BRIS Speed or Beyer for today's race
2. **speed_consistency** = StdDev of last 5 speeds (lower = more consistent)
3. **speed_improvement_trend** = Linear regression slope of last 6 Beyers
4. **best_speed_vs_avg_field** = Horse's top speed - field average
5. **class_par_for_distance** = Expected class at this distance
6. **class_drop_indicator** = (Today's class - avg last 3) / today's class (>0.15 = big drop)
7. **purse_class_ratio** = Today's purse / horse's avg purse last 5

### Pace Features
8. **e1_rating_normalized** = E1 / (distance in furlongs × 10) — standardized
9. **early_speed_points** = How many horses faster E1 in field?
10. **pace_matchup_score** = Likelihood of getting ideal trip given field shape
11. **fractional_pace_projection** = Expected 1st/2nd call fractions
12. **sustained_pace_score** = E2 / E1 ratio (>1.0 = closing speed)

### Form Cycle Features
13. **days_since_peak_speed** = Days from best Beyer last 12mo
14. **form_trend_score** = Weighted average: last 3 races (3× weight on most recent)
15. **bounce_risk** = Recent career-high speed → bounce indicator
16. **improvement_after_claim** = Speed delta if claimed in last race
17. **equipment_change_impact** = Blinkers on/off × historical impact
18. **medication_change_impact** = Lasix add/remove

### Running Style & Trip
19. **running_style_suitability** = Style match vs projected pace
20. **post_position_impact** = Bias-adjusted post advantage/disadvantage
21. **inside_speed_duel_risk** = Multiple E types inside + this horse outside?
22. **closing_room_prob** = Probability of clear stretch run (model-based)
23. **wide_trip_penalty** = Historical track width data

### Pedigree Features
24. **sire_progeny_earnings** = Avg earnings of sire's winners
25. **turf_pedigree_match** = If turf race, use turf pedigree rating
26. **mud_pedigree_match** = If off-track, use mud rating
27. **optimal_distance_score** = Sire avg win dist vs today's distance
28. **dam_produce_quality** = Dam's winners' avg earnings

### Context Features
29. **trainer_angle_flag** = Layoff pattern, turf specialist, etc.
30. **jockey_trainer_synergy** = Combo ROI vs expected
31. **track_meet_timing** = Early/mid/late meet (surface deterioration)
32. **shipping_flag** = Horse's last race different track/region?
33. **field_quality_index** = Avg class rating of all runners
34. **favorite_vulnerability** = ML favorite's weaknesses score

---

## Data Sources Mapping

### Equibase Downloadable Charts (Comma-Delim or XML)
- **Race**: track, date, race#, dist, surface, cond, purse, type, fractions, final time, variant, weather, rail
- **Results**: finish order, beaten lengths, positions at calls, odds, speed figures, comments
- **Payouts**: win/place/show, exotics

### TrackMaster (Enhanced Equibase)
- All Equibase fields PLUS:
- Expanded trip notes, caller comments, trainer/jockey stats

### BRISNET Ultimate PPs
- **Ratings**: BRIS Speed, E1, E2, LP, Class, Prime Power
- **Pedigree**: SPI, DPI, AWD, mud%, turf%, sire/dam stats
- **PP Lines**: 10-12 races with full calls, fractions, ratings
- **Workouts**: Date, distance, time, track, rank (B/Bg/g)
- **Equipment**: Blinkers (B/BF/BE), bandages, Lasix (L)

---

## Critical for Top-5 Prediction

**Must-have features** (Y in Critical column):
- All speed/class ratings (Beyer, BRIS Speed, Class)
- Pace ratings (E1, E2, LP) + running style
- Position at each call (past races + today's result)
- Beaten lengths (granular finish margins)
- Class level today vs historical avg
- Jockey/Trainer stats (current meet, surface-specific)
- Pedigree match (surface, distance, conditions)
- Days rest, equipment changes, layoff patterns
- Track bias adjustments (post, pace, surface)
- Field composition (pace matchup, class spread)

**Nice-to-have** (N in Critical column):
- Weather, ownership, video links, breeder info, show payouts

---

## Storage Format

**Primary**: PostgreSQL (ACID, foreign keys, indexing)
**Analytics**: Parquet files (compressed columnar for fast ML reads)
**Feature Store**: DuckDB (embedded OLAP, fast aggregations)

**Indexes**:
- `races(race_date, track_code)` — time-series queries
- `runners(race_id, horse_id, jockey_id, trainer_id)` — join optimization
- `results(race_id, finish_position)` — label retrieval
- `pp_lines(runner_id, pp_index)` — historical lookups

**Partitioning**: Partition RACES by year (2010-2025 = 16 partitions)

---

## Next Steps
1. **Ingestion Pipeline** (see separate code file)
2. **Feature Engineering** (see separate code file)
3. **Top-5 Ranking Model** (see PyTorch stub)

This schema captures **100% of available data** with zero omissions.
