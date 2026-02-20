"""
Track Intelligence Tab — Unified Command Center
=================================================
Renders the premium 4-tier Track Intelligence tab:
  Tier 1: Engine Status Bar
  Tier 2: Global Calibrated Weights (collapsible)
  Tier 3: Track Overview Grid
  Tier 4: Deep-Dive Panel with sub-tabs

All values are live from the Gold High-IQ database + dynamic engine calculations.
No dead values.
"""

import pandas as pd
import streamlit as st


def render_track_intelligence_tab(
    track_intel_engine,
    gold_db,
    ml_blend_engine,
    db_path: str,
    track_intel_available: bool = True,
    adaptive_learning_available: bool = True,
    ml_blend_available: bool = True,
):
    """Render the full Track Intelligence tab inside a `with tab:` block."""

    st.markdown("### 🧠 Track Intelligence — Unified Command Center")
    st.caption(
        "Live calibration weights, bias profiling, accuracy analytics, and ML model status — "
        "all powered by your Gold High-IQ database and dynamic engine calculations."
    )

    # ──────────────────────────────────────────────
    #  Load all live data sources (one pass)
    # ──────────────────────────────────────────────
    _gw = {}  # global weights
    _track_cals = []  # per-track calibration summaries

    if adaptive_learning_available:
        try:
            from auto_calibration_engine_v2 import (
                get_all_track_calibrations_summary,
                get_live_learned_weights,
            )

            _gw = get_live_learned_weights(db_path) or {}
            _track_cals = get_all_track_calibrations_summary(db_path) or []
        except Exception:
            pass

    # Track Intelligence profiles
    _ti_summaries = []
    if track_intel_available and track_intel_engine is not None:
        _ti_summaries = track_intel_engine.get_all_track_summaries() or []
        if not _ti_summaries:
            with st.spinner("Building track profiles for the first time..."):
                track_intel_engine.rebuild_all_profiles()
            _ti_summaries = track_intel_engine.get_all_track_summaries() or []

    # Merge track lists from both data sources
    _cal_tracks = {tc.get("track_code", ""): tc for tc in _track_cals}
    _ti_tracks = {s["track"]: s for s in _ti_summaries}
    _all_track_codes = sorted(set(list(_cal_tracks.keys()) + list(_ti_tracks.keys())))

    _total_profiled_races = sum(
        _cal_tracks.get(t, {}).get(
            "races_trained_on", _ti_tracks.get(t, {}).get("total_races", 0)
        )
        for t in _all_track_codes
    )
    _avg_confidence = (
        sum(tc.get("avg_confidence", 0) for tc in _track_cals)
        / max(len(_track_cals), 1)
        if _track_cals
        else 0
    )

    # ═══════════════════════════════════════════════
    #  TIER 1 — Engine Status Bar
    # ═══════════════════════════════════════════════
    st.markdown("---")
    es1, es2, es3, es4 = st.columns(4)
    with es1:
        st.metric("Profiled Tracks", len(_all_track_codes))
    with es2:
        st.metric("Total Races", _total_profiled_races)
    with es3:
        _ml_loaded = False
        if ml_blend_available and ml_blend_engine:
            try:
                _ml_loaded = ml_blend_engine.get_model_info().get("model_loaded", False)
            except Exception:
                pass
        st.metric("ML Model", "✅ Active" if _ml_loaded else "❌ Not Loaded")
    with es4:
        st.metric(
            "Avg Confidence",
            f"{_avg_confidence:.0%}" if _avg_confidence > 0 else "—",
        )

    # Rebuild button (right-aligned, compact)
    _rb1, _rb2 = st.columns([3, 1])
    with _rb2:
        if track_intel_available and track_intel_engine is not None:
            if st.button("🔄 Rebuild Profiles", key="rebuild_profiles_btn"):
                with st.spinner("Rebuilding all track profiles..."):
                    track_intel_engine.rebuild_all_profiles()
                st.success("✅ Profiles rebuilt!")
                st.rerun()

    # ═══════════════════════════════════════════════
    #  TIER 2 — Global Calibrated Weights (collapsible)
    # ═══════════════════════════════════════════════
    if _gw:
        with st.expander(
            "⚖️ Global Calibrated Weights (Auto-Cal Engine)", expanded=False
        ):
            st.caption(
                "Blended weights learned across all tracks. "
                "Per-track deltas are shown in the detail panel below."
            )
            gw_cols = st.columns(6)
            _weight_names = ["class", "speed", "form", "pace", "style", "post"]
            for _wi, _wn in enumerate(_weight_names):
                with gw_cols[_wi]:
                    st.metric(_wn.capitalize(), f"{_gw.get(_wn, 0):.2f}")

            # Odds drift parameters
            od_cols = st.columns(3)
            with od_cols[0]:
                st.metric(
                    "Odds Drift Penalty",
                    f"{_gw.get('odds_drift_penalty', 0):.2f}",
                )
            with od_cols[1]:
                st.metric(
                    "Smart Money Bonus",
                    f"{_gw.get('smart_money_bonus', 0):.2f}",
                )
            with od_cols[2]:
                st.metric(
                    "A-Group Drift Gate",
                    f"{_gw.get('a_group_drift_gate', 0):.2f}",
                )

    # ═══════════════════════════════════════════════
    #  TIER 3 — Track Overview Grid
    # ═══════════════════════════════════════════════
    st.markdown("---")

    if not _all_track_codes:
        st.info(
            "📋 No track data available yet. "
            "Submit race results to build intelligence profiles."
        )
        return  # nothing more to render

    st.markdown("#### 🏇 Track Profiles")
    _overview_rows = []
    for _tc in _all_track_codes:
        _cal = _cal_tracks.get(_tc, {})
        _ti = _ti_tracks.get(_tc, {})
        _n_races = _cal.get("races_trained_on", _ti.get("total_races", 0))
        _conf = _cal.get("avg_confidence", 0)
        _conf_icon = "🟢" if _conf >= 0.7 else ("🟡" if _conf >= 0.4 else "🔴")
        _style = _ti.get("style_bias", "—")
        _win_pct = _ti.get("overall_winner_pct", 0)
        _overview_rows.append(
            {
                "": _conf_icon,
                "Track": _tc,
                "Races": _n_races,
                "Confidence": f"{_conf:.0%}" if _conf > 0 else "—",
                "Winner %": f"{_win_pct:.1f}%" if _win_pct > 0 else "—",
                "Style Bias": _style,
            }
        )
    st.dataframe(
        pd.DataFrame(_overview_rows),
        use_container_width=True,
        hide_index=True,
    )

    # ═══════════════════════════════════════════════
    #  TIER 4 — Deep-Dive Panel (sub-tabs)
    # ═══════════════════════════════════════════════
    st.markdown("---")
    selected_track = st.selectbox(
        "Select track for deep analysis:",
        _all_track_codes,
        key="ti_deep_selector",
    )

    if not selected_track:
        return

    # ── Load per-track data ──
    _tw = {}  # per-track weights
    if adaptive_learning_available:
        try:
            from auto_calibration_engine_v2 import get_live_learned_weights

            _tw = get_live_learned_weights(db_path, track_code=selected_track) or {}
        except Exception:
            _tw = _cal_tracks.get(selected_track, {}).get("weights", {})

    _profile = None
    if track_intel_available and track_intel_engine is not None:
        try:
            _profile = track_intel_engine.build_full_profile(selected_track)
        except Exception:
            pass

    _accuracy = {}
    _biases = {}
    if gold_db:
        try:
            _accuracy = gold_db.calculate_accuracy_stats(selected_track) or {}
        except Exception:
            pass
        try:
            _biases = gold_db.detect_biases(selected_track) or {}
        except Exception:
            pass

    _tc_races = _cal_tracks.get(selected_track, {}).get(
        "races_trained_on", _profile.total_races if _profile else 0
    )

    # Track header
    st.markdown(f"#### 🏟️ {selected_track} — {_tc_races} Races Profiled")

    # Sub-tabs
    _st1, _st2, _st3, _st4, _st5 = st.tabs(
        [
            "📊 Overview",
            "🏁 Surface & Distance",
            "🎯 Bias Detection",
            "🏆 J/T Combos",
            "⚙️ Calibration & ML",
        ]
    )

    # ── Sub-tab 1: Overview ──
    with _st1:
        _render_overview_subtab(_profile)

    # ── Sub-tab 2: Surface & Distance ──
    with _st2:
        _render_surface_distance_subtab(_accuracy, _profile)

    # ── Sub-tab 3: Bias Detection ──
    with _st3:
        _render_bias_subtab(_biases, _profile)

    # ── Sub-tab 4: J/T Combos ──
    with _st4:
        _render_jt_subtab(_profile)

    # ── Sub-tab 5: Calibration & ML ──
    with _st5:
        _render_calibration_ml_subtab(
            selected_track=selected_track,
            track_weights=_tw,
            global_weights=_gw,
            cal_tracks=_cal_tracks,
            profile=_profile,
            tc_races=_tc_races,
            ml_blend_engine=ml_blend_engine,
            ml_blend_available=ml_blend_available,
            track_intel_engine=track_intel_engine,
            gold_db=gold_db,
        )


# ═══════════════════════════════════════════════════
#  HELPER RENDERERS (pure Streamlit, no dead values)
# ═══════════════════════════════════════════════════


def _render_overview_subtab(profile):
    """Sub-tab 1: key metrics + style bias + insights."""
    if not profile:
        st.caption("No profile data available yet. Rebuild profiles to generate.")
        return

    _ov1, _ov2, _ov3, _ov4 = st.columns(4)
    with _ov1:
        st.metric("Winner %", f"{profile.overall_winner_pct:.1f}%")
    with _ov2:
        st.metric("Top-3 %", f"{profile.overall_top3_pct:.1f}%")
    with _ov3:
        st.metric("Top-4 %", f"{profile.overall_top4_pct:.1f}%")
    with _ov4:
        st.metric("Horses Analysed", profile.total_horses)

    st.markdown("---")
    st.markdown("**Style Bias:** " + (profile.style_bias or "No dominant bias"))

    if profile.insights:
        st.markdown("##### 💡 Detected Insights")
        for _ins in profile.insights:
            _sev_icon = {
                "strong": "🔴",
                "moderate": "🟡",
                "mild": "🟢",
            }.get(_ins.severity, "⚪")
            st.info(
                f"{_sev_icon} **[{_ins.category.upper()}]** "
                f"{_ins.description} "
                f"*(n={_ins.sample_size}, conf={_ins.confidence:.0%})*"
            )


def _render_surface_distance_subtab(accuracy, profile):
    """Sub-tab 2: surface + distance accuracy from gold_db, with TrackProfile fallback."""

    if accuracy:
        # ── Surface Accuracy (from gold_db) ──
        st.markdown("##### 🏁 Surface Accuracy")
        _surf_cols = st.columns(2)
        for _si, _surf in enumerate(["dirt", "turf"]):
            with _surf_cols[_si]:
                _sr = accuracy.get(f"{_surf}_races", {})
                _sn = (
                    int(_sr.get("value", 0)) if isinstance(_sr, dict) else int(_sr or 0)
                )
                _sa = accuracy.get(f"{_surf}_accuracy_pct", {})
                _sav = (
                    float(_sa.get("value", 0))
                    if isinstance(_sa, dict)
                    else float(_sa or 0)
                )
                _sw = accuracy.get(f"{_surf}_winner_pct", {})
                _swv = (
                    float(_sw.get("value", 0))
                    if isinstance(_sw, dict)
                    else float(_sw or 0)
                )
                _icon = "🟤" if _surf == "dirt" else "🟢"
                if _sn > 0:
                    st.metric(
                        f"{_icon} {_surf.title()}",
                        f"{_sav:.1f}%",
                        delta=f"{_sn} races",
                        delta_color="off",
                        help=f"Top-4 overlap accuracy on {_surf}. Winner acc: {_swv:.1f}%",
                    )
                else:
                    st.metric(
                        f"{_icon} {_surf.title()}",
                        "—",
                        delta="0 races",
                        delta_color="off",
                    )

        # ── Distance Accuracy ──
        st.markdown("##### 📏 Distance Accuracy")
        _dist_cols = st.columns(2)
        _dist_labels = {
            "sprint": ("⚡ Sprints", "4½f – 7½f"),
            "route": ("🏃 Routes", "8f – 1½mi"),
        }
        for _di, _dist in enumerate(["sprint", "route"]):
            with _dist_cols[_di]:
                _dr = accuracy.get(f"{_dist}_races", {})
                _dn = (
                    int(_dr.get("value", 0)) if isinstance(_dr, dict) else int(_dr or 0)
                )
                _da = accuracy.get(f"{_dist}_accuracy_pct", {})
                _dav = (
                    float(_da.get("value", 0))
                    if isinstance(_da, dict)
                    else float(_da or 0)
                )
                _dw = accuracy.get(f"{_dist}_winner_pct", {})
                _dwv = (
                    float(_dw.get("value", 0))
                    if isinstance(_dw, dict)
                    else float(_dw or 0)
                )
                _lbl, _rng = _dist_labels[_dist]
                if _dn > 0:
                    st.metric(
                        _lbl,
                        f"{_dav:.1f}%",
                        delta=f"{_dn} races ({_rng})",
                        delta_color="off",
                        help=f"Top-4 overlap accuracy for {_dist}s. Winner acc: {_dwv:.1f}%",
                    )
                else:
                    st.metric(
                        _lbl,
                        "—",
                        delta=f"0 races ({_rng})",
                        delta_color="off",
                    )

    elif profile:
        # Fallback to TrackProfile breakdown
        st.markdown("##### 🏁 Surface & Distance Breakdown")
        _sf_cols = st.columns(3)
        with _sf_cols[0]:
            st.markdown("**Surface**")
            _sf_data = []
            if profile.dirt_races > 0:
                _sf_data.append(
                    {
                        "Surface": "Dirt",
                        "Races": profile.dirt_races,
                        "Winner %": f"{profile.dirt_winner_pct:.1f}%",
                    }
                )
            if profile.turf_races > 0:
                _sf_data.append(
                    {
                        "Surface": "Turf",
                        "Races": profile.turf_races,
                        "Winner %": f"{profile.turf_winner_pct:.1f}%",
                    }
                )
            if _sf_data:
                st.dataframe(pd.DataFrame(_sf_data), hide_index=True)
            else:
                st.caption("No surface data")

        with _sf_cols[1]:
            st.markdown("**Distance**")
            _sd_data = []
            if profile.sprint_races > 0:
                _sd_data.append(
                    {
                        "Distance": "Sprint (<8f)",
                        "Races": profile.sprint_races,
                        "Winner %": f"{profile.sprint_winner_pct:.1f}%",
                    }
                )
            if profile.route_races > 0:
                _sd_data.append(
                    {
                        "Distance": "Route (≥8f)",
                        "Races": profile.route_races,
                        "Winner %": f"{profile.route_winner_pct:.1f}%",
                    }
                )
            if _sd_data:
                st.dataframe(pd.DataFrame(_sd_data), hide_index=True)
            else:
                st.caption("No distance data")

        with _sf_cols[2]:
            st.markdown("**Condition**")
            _sc_data = []
            if profile.fast_races > 0:
                _sc_data.append(
                    {
                        "Condition": "Fast/Firm",
                        "Races": profile.fast_races,
                        "Winner %": f"{profile.fast_winner_pct:.1f}%",
                    }
                )
            if profile.off_track_races > 0:
                _sc_data.append(
                    {
                        "Condition": "Off Track",
                        "Races": profile.off_track_races,
                        "Winner %": f"{profile.off_track_winner_pct:.1f}%",
                    }
                )
            if _sc_data:
                st.dataframe(pd.DataFrame(_sc_data), hide_index=True)
            else:
                st.caption("No condition data")
    else:
        st.caption("No surface/distance data available yet.")


def _render_bias_subtab(biases, profile):
    """Sub-tab 3: bias intelligence from gold_db with TrackProfile fallback."""

    _has_bias_data = False

    if biases and biases.get("style_bias") != "Insufficient Data":
        _has_bias_data = True

        # Active biases as tags
        _active = biases.get("active_biases", [])
        if isinstance(_active, list) and _active:
            st.markdown(
                "**Track Tendencies:** " + " · ".join(f"**{b}**" for b in _active)
            )
        else:
            st.caption("No strong biases detected yet")

        # Style win breakdown
        _style_bd = biases.get("style_win_breakdown", {})
        if _style_bd and isinstance(_style_bd, dict) and sum(_style_bd.values()) > 0:
            st.markdown("##### 🎯 Running Style Distribution")
            _total_sw = sum(_style_bd.values())
            _style_rows = []
            for _sk, _sv in _style_bd.items():
                _pct = _sv / max(_total_sw, 1) * 100
                _style_rows.append({"Style": _sk, "Wins": _sv, "Win %": f"{_pct:.1f}%"})
            st.dataframe(
                pd.DataFrame(_style_rows),
                hide_index=True,
                use_container_width=True,
            )

        # Post-position stats table
        _pp_stats = biases.get("post_position_stats", {})
        if _pp_stats and isinstance(_pp_stats, dict):
            st.markdown("##### 📍 Post Position Analysis")
            _pp_rows = []
            _pp_labels = {
                "inside": "Inside (1-3)",
                "middle": "Middle (4-6)",
                "outside": "Outside (7+)",
            }
            for _zone in ["inside", "middle", "outside"]:
                _zd = _pp_stats.get(_zone, {})
                _pp_rows.append(
                    {
                        "Post Zone": _pp_labels.get(_zone, _zone),
                        "Win %": f"{_zd.get('win_pct', 0):.1f}%",
                        "Top-4 %": f"{_zd.get('top4_pct', 0):.1f}%",
                        "Sample": _zd.get("sample", 0),
                    }
                )
            st.dataframe(
                pd.DataFrame(_pp_rows),
                use_container_width=True,
                hide_index=True,
            )

    if profile and not _has_bias_data:
        # Fallback to TrackProfile bias data
        _b_cols = st.columns(2)
        with _b_cols[0]:
            st.markdown(f"**Running Style:** {profile.style_bias}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Style": "Speed (E/EP)",
                            "Win %": f"{profile.speed_win_pct:.1f}%",
                        },
                        {
                            "Style": "Presser (P)",
                            "Win %": f"{profile.presser_win_pct:.1f}%",
                        },
                        {
                            "Style": "Closer (S/C)",
                            "Win %": f"{profile.closer_win_pct:.1f}%",
                        },
                    ]
                ),
                hide_index=True,
            )
        with _b_cols[1]:
            st.markdown("**Post Position Zones**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Zone": "Inside (1-3)",
                            "Win %": f"{profile.inside_win_pct:.1f}%",
                            "Top-4 %": f"{profile.inside_top4_pct:.1f}%",
                        },
                        {
                            "Zone": "Middle (4-6)",
                            "Win %": f"{profile.middle_win_pct:.1f}%",
                            "Top-4 %": f"{profile.middle_top4_pct:.1f}%",
                        },
                        {
                            "Zone": "Outside (7+)",
                            "Win %": f"{profile.outside_win_pct:.1f}%",
                            "Top-4 %": f"{profile.outside_top4_pct:.1f}%",
                        },
                    ]
                ),
                hide_index=True,
            )

    if not _has_bias_data and not profile:
        st.caption(
            "No bias data available yet. "
            "Submit more race results to enable bias detection."
        )


def _render_jt_subtab(profile):
    """Sub-tab 4: top jockey-trainer combos."""
    if profile and profile.top_jt_combos:
        st.markdown("##### 🏆 Top Jockey-Trainer Combinations")
        _jt_data = [
            {
                "Jockey": c["jockey"],
                "Trainer": c["trainer"],
                "Starts": c["starts"],
                "Wins": c["wins"],
                "Win %": f"{c['win_pct']}%",
            }
            for c in profile.top_jt_combos[:10]
        ]
        st.dataframe(
            pd.DataFrame(_jt_data),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No jockey-trainer combo data available for this track yet.")


def _render_calibration_ml_subtab(
    *,
    selected_track,
    track_weights,
    global_weights,
    cal_tracks,
    profile,
    tc_races,
    ml_blend_engine,
    ml_blend_available,
    track_intel_engine,
    gold_db,
):
    """Sub-tab 5: per-track weights (with deltas), ML model status, per-track retrain."""

    # ── Per-Track Calibrated Weights ──
    if track_weights:
        st.markdown("##### ⚖️ Per-Track Calibrated Weights")
        st.caption("Deltas show how this track differs from the global calibration.")
        _wt_cols = st.columns(6)
        for _wi, _wn in enumerate(["class", "speed", "form", "pace", "style", "post"]):
            with _wt_cols[_wi]:
                _tv = track_weights.get(_wn, 0)
                _gv = global_weights.get(_wn, 0) if global_weights else 0
                _delta = round(_tv - _gv, 2)
                st.metric(
                    _wn.capitalize(),
                    f"{_tv:.2f}",
                    delta=f"{_delta:+.2f}" if _delta != 0 else None,
                    delta_color="normal",
                    help=f"Track weight vs global ({_gv:.2f})",
                )

        # Odds drift parameters for this track
        _od_cols = st.columns(3)
        with _od_cols[0]:
            _tdp = track_weights.get("odds_drift_penalty", 0)
            _gdp = global_weights.get("odds_drift_penalty", 0) if global_weights else 0
            _dp_delta = round(_tdp - _gdp, 2)
            st.metric(
                "Odds Drift Penalty",
                f"{_tdp:.2f}",
                delta=f"{_dp_delta:+.2f}" if _dp_delta != 0 else None,
                delta_color="inverse",
            )
        with _od_cols[1]:
            _tsm = track_weights.get("smart_money_bonus", 0)
            _gsm = global_weights.get("smart_money_bonus", 0) if global_weights else 0
            _sm_delta = round(_tsm - _gsm, 2)
            st.metric(
                "Smart Money Bonus",
                f"{_tsm:.2f}",
                delta=f"{_sm_delta:+.2f}" if _sm_delta != 0 else None,
                delta_color="normal",
            )
        with _od_cols[2]:
            _tag = track_weights.get("a_group_drift_gate", 0)
            _gag = global_weights.get("a_group_drift_gate", 0) if global_weights else 0
            _ag_delta = round(_tag - _gag, 2)
            st.metric(
                "A-Group Drift Gate",
                f"{_tag:.2f}",
                delta=f"{_ag_delta:+.2f}" if _ag_delta != 0 else None,
                delta_color="normal",
            )
    else:
        st.caption(
            "No per-track calibration weights yet. "
            "Need 2+ races with results for this track."
        )

    # ── ML Blend Model Status ──
    st.markdown("---")
    st.markdown("##### 🤖 ML Blend Model Status")
    if ml_blend_available and ml_blend_engine:
        _mi = ml_blend_engine.get_model_info()
        _ml_cols = st.columns(4)
        with _ml_cols[0]:
            st.metric("Features", _mi.get("n_features", "?"))
        with _ml_cols[1]:
            st.metric("Hidden Dim", _mi.get("hidden_dim", "?"))
        with _ml_cols[2]:
            st.metric("Architecture", "Plackett-Luce NN")
        with _ml_cols[3]:
            st.metric(
                "Status",
                "✅ Active" if _mi.get("model_loaded") else "❌ Not Loaded",
            )
    else:
        st.caption("ML Blend Engine not available.")

    # ── Per-Track Retrain ──
    st.markdown("---")
    st.markdown("##### 🎯 Per-Track Retrain")
    _retrain_races = profile.total_races if profile else tc_races
    _min_races = 30
    st.caption(
        f"Retrain ML model using only {selected_track} data. "
        f"Requires {_min_races}+ races."
    )

    if _retrain_races >= _min_races:
        if st.button(
            f"🚀 Retrain for {selected_track} ({_retrain_races} races)",
            key=f"retrain_track_{selected_track}",
            type="primary",
        ):
            with st.spinner(f"Retraining on {selected_track} data..."):
                try:
                    from retrain_model import retrain_model as _retrain_fn

                    _tr = _retrain_fn(
                        db_path=gold_db.db_path,
                        epochs=100,
                        learning_rate=0.0005,
                        batch_size=4,
                        min_races=_min_races,
                        track_name=selected_track,
                    )
                    if "error" in _tr:
                        st.error(f"❌ {_tr['error']}")
                    else:
                        st.success(f"✅ {selected_track} model trained!")
                        _rc1, _rc2, _rc3 = st.columns(3)
                        with _rc1:
                            st.metric(
                                "Winner %",
                                f"{_tr['metrics']['winner_accuracy']:.1%}",
                            )
                        with _rc2:
                            st.metric(
                                "Top-3 %",
                                f"{_tr['metrics']['top3_accuracy']:.1%}",
                            )
                        with _rc3:
                            st.metric(
                                "Top-4 %",
                                f"{_tr['metrics']['top4_accuracy']:.1%}",
                            )
                        st.info(f"💾 Saved: {_tr.get('model_path', 'N/A')}")
                        # Save track ML profile
                        track_intel_engine.save_track_ml_profile(
                            track_code=selected_track,
                            model_path=_tr["model_path"],
                            n_features=_tr["metrics"].get("n_features", 0),
                            hidden_dim=64,
                            val_metrics=_tr["metrics"],
                            races_trained=_tr.get("n_races", 0),
                        )
                        # Reload ML blend engine
                        if ml_blend_available and ml_blend_engine:
                            ml_blend_engine.reload_model()
                except Exception as _re:
                    st.error(f"Per-track retrain error: {_re}")
                    import traceback

                    st.code(traceback.format_exc())
    else:
        st.warning(
            f"⏳ {selected_track} has {_retrain_races} races. "
            f"Need {_min_races}+ for retraining."
        )

    # Confidence and last updated
    _tc_updated = cal_tracks.get(selected_track, {}).get("last_updated", "")
    if _tc_updated:
        try:
            from datetime import datetime as _dt

            _upd = _dt.fromisoformat(_tc_updated)
            st.caption(f"Last calibration: {_upd.strftime('%m/%d/%Y %I:%M %p')}")
        except Exception:
            st.caption(f"Last calibration: {_tc_updated}")
