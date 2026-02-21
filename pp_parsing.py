"""
PP Parsing Module - BRISNET Past Performance Data Parser
=========================================================
Phase 3 extraction from app.py (Feb 21, 2026)

Contains all functions for parsing BRISNET past performance text data:
- Race header parsing (track, surface, distance, conditions)
- Horse data extraction (styles, speed figures, angles, pedigree)
- Bias/impact value parsing (track bias, weekly post bias, SPI)
- Race history and workout parsing
- Race metadata extraction

All functions are pure: they take text input and return structured data.
No Streamlit dependencies. No rating calculations.
"""

import contextlib
import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import TRACK_ALIASES, _CANON_BY_TOKEN
from utils import _normalize_style

logger = logging.getLogger(__name__)



def _find_header_line(pp_text: str) -> str:
    """
    Find the actual BRISNET header line by skipping copyright/preamble lines.

    BRISNET PPs often start with lines like:
      (c) Copyright 2026 BRIS...
      (blank lines)
    before the real header:
      Ultimate PP's w/ QuickPlay Comments | Oaklawn Park | ...
    or:
      Ultimate PP's w/ QuickPlay Comments Oaklawn Park Alw 12500s ...
    """
    if not pp_text:
        return ""
    lines = pp_text.strip().split("\n")
    for line in lines[:15]:  # Check first 15 lines for the header
        stripped = line.strip()
        if not stripped:
            continue
        # Skip copyright lines: "(c) Copyright", "©", "Copyright"
        if stripped.startswith("(c)") or stripped.lower().startswith("copyright"):
            continue
        # Skip lines that are only special chars / very short garbage
        if len(stripped) < 5:
            continue
        # Skip lines that look like disclaimers or legal boilerplate
        if any(
            kw in stripped.lower()
            for kw in [
                "all rights reserved",
                "unauthorized",
                "license",
                "brisnet.com/terms",
            ]
        ):
            continue
        # This is the first real content line — use it as the header
        return stripped
    # Fallback: just use the first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    return ""



def parse_track_name_from_pp(pp_text: str) -> str:
    """
    Parse track name from BRISNET PP text.
    Checks both:
    1. Header/title text for full track names
    2. Race history lines for track abbreviations (e.g., 29Dec25Tup, 08Nov25Aquª)
    """
    text = (pp_text or "")[:2000].lower()  # Increased to capture race history

    # First, check for track abbreviations in race history date lines
    # Pattern: DDMmmYYTrk (e.g., 29Dec25Tup, 08Nov25Aquª)
    date_line_pattern = r"\d{2}[A-Za-z]{3}\d{2}([A-Za-z]{2,4})"
    for match in re.finditer(date_line_pattern, text):
        track_code = match.group(1).lower()
        if track_code in _CANON_BY_TOKEN:
            return _CANON_BY_TOKEN[track_code]

    # Second, check for full track names in header
    for token, canon in _CANON_BY_TOKEN.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return canon

    # Third, check for multi-word track names
    for canon, toks in TRACK_ALIASES.items():
        for t in toks:
            t_words = [w for w in t.split() if len(w) > 2]
            if t_words and all(
                re.search(rf"\b{re.escape(w)}\b", text) for w in t_words
            ):
                return canon

    logger.debug("parse_track_name_from_pp: no track detected in first 2000 chars")
    return ""



def detect_race_number(pp_text: str) -> int | None:
    """Extract race number from PP text header (e.g., 'Race 6')."""
    s = pp_text or ""
    if not s:
        logger.debug("detect_race_number: empty pp_text")
        return None
    # Look for "Race N" pattern in first few lines
    m = re.search(r"(?mi)\bRace\s+(\d+)\b", s[:500])
    if m:
        try:
            return int(m.group(1))
        except (ValueError, AttributeError):
            # Regex group missing or invalid
            pass
    return None



def parse_brisnet_race_header(pp_text: str) -> dict[str, Any]:
    """
    Parse comprehensive BRISNET race header information.

    Expected format:
    Ultimate PP's w/ QuickPlay Comments | Track Name | Race Type Purse | Distance | Age/Sex | Date | Race #

    Example:
    Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8

    Returns dict with all extracted fields:
    - track_name, race_number, race_type, purse_amount, distance, age_restriction, sex_restriction, race_date, day_of_week
    """
    if not pp_text:
        logger.debug("parse_brisnet_race_header: empty pp_text, returning {}")
        return {}

    result = {
        "track_name": "",
        "race_number": 0,
        "race_type": "",
        "purse_amount": 0,
        "distance": "",
        "age_restriction": "",
        "sex_restriction": "",
        "race_date": "",
        "day_of_week": "",
    }

    # Get the actual header line (skip copyright/preamble lines)
    header_line = _find_header_line(pp_text)

    # Split by pipe delimiter
    parts = [p.strip() for p in header_line.split("|")]

    if len(parts) < 2:
        # ===== Non-pipe format parser =====
        # Format: "Ultimate PP's w/ QuickPlay Comments Oaklawn Park Alw 12500s 6 Furlongs 4&up Thursday, February 05, 2026 Race 9"
        text = header_line

        # Strip product-type prefixes:
        #   "Ultimate PP's w/ QuickPlay Comments"
        #   "Premium Plus PP's"
        #   and similar BRISNET product headers
        text = re.sub(
            r"^(?:Ultimate\s+PP.*?Comments|Premium\s+Plus\s+PP[''']?s?(?:\s+MC)?)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # --- Extract Race Number (anchored at end, e.g., "Race 9") ---
        race_num_match = re.search(r"\bRace\s+(\d+)\s*$", text, re.IGNORECASE)
        if race_num_match:
            with contextlib.suppress(ValueError, TypeError):
                result["race_number"] = int(race_num_match.group(1))
            text = text[: race_num_match.start()].strip()

        # --- Extract Date (e.g., "Thursday, February 05, 2026") ---
        date_pattern = r"((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4})"
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            comma_parts = date_str.split(",", 1)
            if len(comma_parts) >= 2:
                result["day_of_week"] = comma_parts[0].strip()
                result["race_date"] = comma_parts[1].strip()
            text = text[: date_match.start()].strip().rstrip(",").strip()

        # --- Extract Distance (e.g., "6 Furlongs", "1 1/16 Miles", "1„ Mile") ---
        dist_pattern = r"([\d½¼¾„ˆ]+(?:\s*[\d½¼¾/]+)?\s*(?:Furlongs?|Miles?|Yards?)\b)"
        dist_match = re.search(dist_pattern, text, re.IGNORECASE)
        if dist_match:
            result["distance"] = dist_match.group(1).strip()
            text = (
                text[: dist_match.start()].strip()
                + " "
                + text[dist_match.end() :].strip()
            )
            text = text.strip()

        # --- Extract Age/Sex restrictions (e.g., "4&up", "3yo Fillies", "F&M") ---
        age_sex_pattern = r"\b(\d+&up|3yo|4yo|F&M|Fillies|Mares|Colts|Geldings|C&G)\b"
        age_sex_matches = re.findall(age_sex_pattern, text, re.IGNORECASE)
        for m in age_sex_matches:
            ml = m.lower()
            if re.match(r"\d+&up|\d+yo", ml):
                result["age_restriction"] = m
            elif re.match(r"fillies?|mares?|colts?|geldings?|f&m|c&g", ml):
                result["sex_restriction"] = m.title()
        # Remove matched age/sex tokens from text
        text = re.sub(age_sex_pattern, "", text, flags=re.IGNORECASE).strip()

        # --- Extract Track Name ---
        # Strategy 1: Match against known track names (longest match first for accuracy)
        text_lower = text.lower().strip()
        matched_track = ""
        matched_len = 0
        for canon, toks in TRACK_ALIASES.items():
            for t in toks:
                if text_lower.startswith(t) and len(t) > matched_len:
                    # Verify word boundary (not a partial match)
                    if len(text_lower) == len(t) or not text_lower[len(t)].isalpha():
                        matched_track = canon
                        matched_len = len(t)
        if matched_track:
            result["track_name"] = matched_track
            text = text[matched_len:].strip()
        else:
            # Strategy 2: Everything before the first race-type keyword is the track name
            race_type_boundary = re.search(
                r"\b(Alw|Clm|Stk|Hcp|Mdn|Msw|Aoc|Mcl|Wmc|Oc|Soc|Str|G1|G2|G3|PWC|"
                r"©|¨|§|Allowance|Claiming|Stakes|Handicap|Maiden|Optional)\b",
                text,
                re.IGNORECASE,
            )
            if race_type_boundary:
                candidate = text[: race_type_boundary.start()].strip()
                if candidate and len(candidate) > 2:
                    result["track_name"] = candidate
                    text = text[race_type_boundary.start() :].strip()
            else:
                # Strategy 3: Everything before any digit sequence is the track name
                digit_boundary = re.search(r"\b\d", text)
                if digit_boundary and digit_boundary.start() > 2:
                    result["track_name"] = text[: digit_boundary.start()].strip()
                    text = text[digit_boundary.start() :].strip()

        # --- Extract Race Type + Purse from remaining text ---
        # e.g., "Alw 12500s", "Clm 25000", "©Hcp 50000", "Str 40000"
        type_purse_match = re.search(r"([©¨§]?\w+)\s+(\d+)\w*", text)
        if type_purse_match:
            result["race_type"] = type_purse_match.group(1)
            with contextlib.suppress(ValueError, TypeError):
                result["purse_amount"] = int(type_purse_match.group(2))

        return result

    # Parse each section
    for i, part in enumerate(parts):
        part_lower = part.lower()

        # Skip product-type segments like "Ultimate PP's w/ QuickPlay Comments" or "Premium Plus PP's"
        if (
            "ultimate" in part_lower
            or "quickplay" in part_lower
            or "premium plus" in part_lower
        ):
            continue

        # Track name (first non-Ultimate part)
        if not result["track_name"] and i > 0:
            # Check if this looks like a track name
            if (
                not re.search(r"\d+\s*(?:furlong|mile|yard)", part_lower)
                and not re.search(r"(clm|alw|stk|hcp|mdn|msw|aoc)", part_lower)
                and not re.search(r"race\s+\d+", part_lower)
                and not re.search(
                    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                    part_lower,
                )
            ):
                result["track_name"] = part
                continue

        # Race type + purse (e.g., "©Hcp 50000", "Clm 4000")
        race_type_match = re.search(r"([©¨§]?[A-Za-z]+)\s+(\d+)", part)
        if race_type_match and not result["race_type"]:
            result["race_type"] = race_type_match.group(1)
            with contextlib.suppress(Exception):
                result["purse_amount"] = int(race_type_match.group(2))
            continue

        # Distance (e.g., "6 Furlongs", "1 Mile", "1„ Mile")
        if re.search(r"\d+\s*(?:furlong|mile|yard|f\b)", part_lower):
            result["distance"] = part
            continue

        # Age/Sex restrictions (e.g., "3yo Fillies", "4&up", "F&M")
        if re.search(
            r"(?:\d+yo|f&m|fillies|mares|colts|geldings|4&up|3&up)", part_lower
        ):
            age_match = re.search(r"(\d+yo|4&up|3&up)", part_lower)
            if age_match:
                result["age_restriction"] = age_match.group(1)

            sex_match = re.search(
                r"(fillies?|mares?|colts?|geldings?|f&m)", part_lower, re.IGNORECASE
            )
            if sex_match:
                result["sex_restriction"] = sex_match.group(1).title()
            continue

        # Date (e.g., "Monday, February 02, 2026")
        date_match = re.search(
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(.+\d{4})",
            part_lower,
        )
        if date_match:
            result["day_of_week"] = date_match.group(1).title()
            result["race_date"] = date_match.group(2).strip()
            continue

        # Race number (e.g., "Race 8")
        race_num_match = re.search(r"race\s+(\d+)", part_lower)
        if race_num_match:
            with contextlib.suppress(Exception):
                result["race_number"] = int(race_num_match.group(1))

    return result



def detect_race_type(pp_text: str) -> str:
    """
    Normalize many wordings into the exact key set used by base_class_bias.
    """
    s = (pp_text or "")[:1000].lower()

    # Graded stakes first
    if re.search(r"\b(g1|grade\s*i)\b", s):
        return "stakes (g1)"
    if re.search(r"\b(g2|grade\s*ii)\b", s):
        return "stakes (g2)"
    if re.search(r"\b(g3|grade\s*iii)\b", s):
        return "stakes (g3)"

    # Listed / generic stakes
    if "listed" in s:
        return "stakes (listed)"
    if re.search(r"\bstakes?\b", s):
        return "stakes"

    # Maiden
    # Check for standalone "MC" followed by number (e.g., "MC 16000" or "MC16000")
    if re.search(r"\bmc\s*\d", s):
        return "maiden claiming"

    if re.search(r"\b(mdn|maiden)\b", s):
        if re.search(r"(mcl|mdn\s*clm|maiden\s*claim)", s):
            return "maiden claiming"
        if re.search(r"(msw|maiden\s*special|maiden\s*sp\s*wt)", s):
            return "maiden special weight"
        return "maiden special weight"

    # Starter Optional Claiming (check before general OC/AOC)
    if re.search(r"\b(soc|starter\s*optional\s*claim\w*)\b", s):
        return "starter optional claiming"

    # AOC
    if re.search(r"\b(oc|aoc|optional\s*claim)\b", s):
        return "allowance optional claiming (aoc)"

    # Starter
    if re.search(r"\bstarter\s*allow", s):
        return "starter allowance"
    if re.search(r"\bstarter\s*h(andi)?cap\b", s):
        return "starter handicap"

    # Waiver Claiming
    if re.search(r"\b(waiver|wcl|w\s*clm)\b", s):
        return "waiver claiming"

    # Claiming
    if re.search(r"\bclm|claiming\b", s):
        return "claiming"

    # Allowance last
    if re.search(r"\ballow(ance)?\b", s):
        return "allowance"

    return "allowance"



def _post_bucket(post_str: str) -> str:
    try:
        post = int(re.sub(r"[^\d]", "", str(post_str)))
    except Exception as e:
        st.warning(f"Failed to parse post number: '{post_str}'. Error: {e}")
        post = None  # Default to None and let it become 'mid'
    if post is None:
        return "mid"
    if post == 1:
        return "rail"
    if 2 <= post <= 3:
        return "inner"
    if 4 <= post <= 7:
        return "mid"
    return "outside"

HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
    (?:POST\s+)?          # optional "POST " prefix
    (\d+[A-Za-z]?)        # program number (may have letter suffix for coupled entries, e.g. "1A")
    (?:pp(\d+))?          # optional "pp" + post position (e.g. "pp6" in "1pp6")
    [:\s]+                # colon or whitespace separator
    ([A-Za-z0-9''\u2019.\-\s&]+?)  # horse name (lazy to avoid over-capturing)
    [^\w(]*               # skip any special chars/icons between name and parentheses
    \(\s*
    (E\/P|EP|E|P|S|NA)   # running style
    (?:\s+(\d+))?         # optional quirin
    \s*\)                 # closing paren
    (?:\s*-\s*ML\s+\S+)?  # optional "- ML odds" suffix
    (?:\s*\*+\s*.+)?       # optional "*** ACTUAL WINNER ***" annotation
    .*$                   # allow trailing special chars/icons (BRISNET symbols like ì)
    """,
    re.VERBOSE,
)



def calculate_style_strength(style: str, quirin: float) -> str:
    """Calculate style strength label from running style + Quirin Speed Points.

    For NA (unknown) styles, use QSP to infer strength instead of defaulting to 'Solid':
    - QSP >= 6: 'Solid' (decent early speed signal from BRISNET)
    - QSP 3-5: 'Slight' (some signal but inconclusive)
    - QSP 0-2 or NaN: 'Weak' (no meaningful data to differentiate)
    """
    s = (style or "NA").upper()
    try:
        q = float(quirin)
    except Exception:
        return "Weak" if s == "NA" else "Solid"
    if pd.isna(q):
        return "Weak" if s == "NA" else "Solid"
    # NA style: use QSP to infer strength (Feb 11, 2026)
    if s == "NA":
        if q >= 6:
            return "Solid"
        if q >= 3:
            return "Slight"
        return "Weak"
    if s in ("E", "E/P"):
        if q >= 7:
            return "Strong"
        if q >= 5:
            return "Solid"
        if q >= 3:
            return "Slight"
        return "Weak"
    if s in ("P", "S"):
        if q >= 5:
            return "Slight"
        if q >= 3:
            return "Solid"
        return "Strong"
    return "Solid"



def split_into_horse_chunks(pp_text: str) -> list[tuple]:
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pp_text)
        # group(2) is the post from "ppN"; group(1) is program# (fallback when pp absent)
        post = (m.group(2) or m.group(1)).strip()
        name = m.group(3).strip()
        block = pp_text[start:end]
        chunks.append((post, name, block))
    return chunks



def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows = []
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        program_num = m.group(1).strip()
        # group(2) is the post from "ppN"; group(1) is program# (fallback when pp absent)
        post = (m.group(2) or program_num).strip()
        name = m.group(3).strip()
        style = _normalize_style(m.group(4))
        qpts = m.group(5)
        quirin = int(qpts) if qpts else np.nan
        auto_strength = calculate_style_strength(style, quirin)
        rows.append(
            {
                "#": program_num,
                "Post": post,
                "Horse": name,
                "DetectedStyle": style,
                "Quirin": quirin,
                "AutoStrength": auto_strength,
                "OverrideStyle": "",
                "StyleStrength": auto_strength,
            }
        )
    seen = set()
    uniq = []
    for r in rows:
        key = (r["#"], r["Horse"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    df = pd.DataFrame(uniq)
    if not df.empty:
        df["Quirin"] = df["Quirin"].clip(lower=0, upper=8)
    return df

_ODDS_TOKEN = r"(\d+\s*/\s*\d+|\d+\s*-\s*\d+|[+-]?\d+(?:\.\d+)?)"

# ========================================================


def extract_morning_line_by_horse(pp_text: str) -> dict[str, str]:
    ml = {}
    blocks = {name: block for _, name, block in split_into_horse_chunks(pp_text)}
    for name, block in blocks.items():
        if name in ml:
            continue
        m_start = re.search(rf"(?mi)^\s*{_ODDS_TOKEN}", block or "")
        if m_start:
            ml[name.strip()] = m_start.group(1).replace(" ", "")
            continue
        m_labeled = re.search(
            rf"(?mi)^.*?\b(?:M/?L|Morning\s*Line|ML)\b.*?{_ODDS_TOKEN}", block or ""
        )
        if m_labeled:
            ml[name.strip()] = m_labeled.group(1).replace(" ", "")
    return ml

ANGLE_LINE_RE = re.compile(
    r"""(?mix)^\s*\+?           # optional + prefix (positive trend indicator)
    (                              # ── category capture group ──
      \d{4}                        # year-based stats  (2025, 2024 …)
    | JKYw/\s*(?:                  # jockey angle variants
        Sprints                    #   JKYw/ Sprints
      | Routes                     #   JKYw/ Routes
      | Trn\s*L(?:30|45|60)        #   JKYw/ Trn L30/L45/L60
      | (?:[EPS]|NA)\s*types       #   JKYw/ S types, E types, P types, NA types
      )
    | 1st\s*time\s*(?:str|Turf|Dirt|AW)  # first time surface / distance
    | Debut\s*MdnSpWt              # debut maiden special weight
    | Maiden\s*Sp\s*Wt             # maiden special weight
    | 2nd\s*career\s*race          # 2nd career race
    | Turf\s*to\s*Dirt             # surface switch
    | Dirt\s*to\s*Turf             # surface switch
    | Rte\s*to\s*Sprint            # distance switch
    | Sprint\s*to\s*Rte            # distance switch
    | MdnClm\s*to\s*Mdn            # class change
    | Shipper                      # shipper
    | Blinkers\s*(?:on|off)        # equipment change
    | (?:\d+(?:-\d+)?)\s*days?Away # layoff
    | Sprints                      # standalone distance
    | Routes                       # standalone distance
    | Dirt\s*starts                # standalone surface
    | Turf\s*starts                # standalone surface
    )\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$
    """
)



def parse_angles_for_block(block) -> pd.DataFrame:
    rows = []
    if not block:
        return pd.DataFrame(rows)
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in ANGLE_LINE_RE.finditer(block_str):
        cat, starts, win, itm, roi = m.groups()
        rows.append(
            {
                "Category": re.sub(r"\s+", " ", cat.strip()),
                "Starts": int(starts),
                "Win%": float(win),
                "ITM%": float(itm),
                "ROI": float(roi),
            }
        )
    return pd.DataFrame(rows)



def parse_pedigree_snips(block) -> dict:
    out = {
        "sire_awd": np.nan,
        "sire_1st": np.nan,
        "sire_mud_pct": np.nan,  # %Mud from Sire Stats (Feb 13, 2026)
        "damsire_awd": np.nan,
        "damsire_1st": np.nan,
        "dam_dpi": np.nan,
    }
    if not block:
        return out
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    s = re.search(
        r"(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi",
        block_str,
    )
    if s:
        out["sire_awd"] = float(s.group(1))
        out["sire_mud_pct"] = float(s.group(2))  # Group 2 is %Mud
        out["sire_1st"] = float(s.group(3))  # Group 3 is %-1st
    ds = re.search(
        r"(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi",
        block_str,
    )
    if ds:
        out["damsire_awd"] = float(ds.group(1))
        out["damsire_1st"] = float(ds.group(3))
    d = re.search(r"(?mi)^\s*Dam:\s*DPI\s*(\d+(?:\.\d+)?)\s+(\d+)%", block_str)
    if d:
        out["dam_dpi"] = float(d.group(1))

    # Parse Pedigree breeding ratings: "Pedigree: Fast 85 Off 72 Dist 90 Turf 78"
    # (Feb 13, 2026 enhancement for surface-switch pedigree integration)
    ped_ratings = re.search(
        r"(?mi)Pedigree:\s*Fast\s+(\d+)\s+Off\s+(\d+)\s+Dist\s+(\d+)\s+Turf\s+(\d+)",
        block_str,
    )
    if ped_ratings:
        out["pedigree_fast"] = float(ped_ratings.group(1))
        out["pedigree_off"] = float(ped_ratings.group(2))
        out["pedigree_distance"] = float(ped_ratings.group(3))
        out["pedigree_turf"] = float(ped_ratings.group(4))

    return out



# ========== SAVANT-LEVEL ENHANCEMENTS (Jan 2026) ==========


def parse_claiming_prices(block) -> list[int]:
    """Extract claiming prices from race lines. Returns list of prices (most recent first)."""
    prices = []
    if not block:
        logger.debug("parse_claiming_prices: empty block")
        return prices
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r"Clm\s+(\d+)", block_str):
        with contextlib.suppress(BaseException):
            prices.append(int(m.group(1)))
    return prices[:5]



def detect_lasix_change(block) -> float:
    """SAVANT ANGLE: Lasix/medication changes. Returns bonus from -0.12 to +0.18"""
    bonus = 0.0
    if not block:
        logger.debug("detect_lasix_change: empty block")
        return bonus
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    race_lines = [
        line
        for line in block_str.split("\n")
        if re.search(r"\d{2}[A-Za-z]{3}\d{2}", line)
    ]
    lasix_pattern = []
    for line in race_lines[:5]:
        if re.search(r"\s+L\s+\d+\.\d+\s*$", line):
            lasix_pattern.append(True)
        else:
            lasix_pattern.append(False)
    # CRITICAL FIX: Validate list has at least 2 elements before accessing indices
    if len(lasix_pattern) >= 2:
        if lasix_pattern[0] and not any(lasix_pattern[1:]):
            bonus += 0.18  # First-time Lasix = major boost
        elif not lasix_pattern[0] and lasix_pattern[1]:
            bonus -= 0.12  # Lasix off = red flag
        elif lasix_pattern[0] and sum(lasix_pattern) >= 3:
            bonus += 0.02  # Consistent user
    return bonus



def parse_fractional_positions(block) -> list[list[int]]:
    """Extract running positions: PP, Start, 1C, 2C, Stretch, Finish."""
    positions = []
    if not block:
        logger.debug("parse_fractional_positions: empty block")
        return positions
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # More flexible pattern with unicode range for position markers (¹²³ª etc)
    pattern = r"(\d{2}[A-Za-z]{3}\d{2}).*?(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})"
    for m in re.finditer(pattern, block_str, re.MULTILINE):
        try:
            # Validate each group exists and contains digits before conversion
            pos = []
            for i in range(2, 8):
                group_val = m.group(i)
                if group_val and group_val.isdigit():
                    pos.append(int(group_val))
                else:
                    # Invalid group - skip this match
                    raise ValueError(f"Invalid position group {i}: {group_val}")
            if len(pos) == 6:  # Ensure we got all 6 positions
                positions.append(pos)
        except (ValueError, IndexError):
            # Log parsing failures for debugging (silent failures hide bugs)
            pass  # Could add logging here if needed
    return positions[:5]



def parse_e1_e2_lp_values(block) -> dict:
    """Extract E1, E2, and LP pace figures."""
    e1_vals, e2_vals, lp_vals = [], [], []
    if not block:
        logger.debug("parse_e1_e2_lp_values: empty block")
        return {"e1": e1_vals, "e2": e2_vals, "lp": lp_vals}
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r"(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})", block_str):
        try:
            e1_vals.append(int(m.group(1)))
            e2_vals.append(int(m.group(2)))
            lp_vals.append(int(m.group(3)))
        except (ValueError, AttributeError, IndexError):
            # Regex group missing or conversion failed
            pass
    return {"e1": e1_vals[:5], "e2": e2_vals[:5], "lp": lp_vals[:5]}



# ========== END SAVANT ENHANCEMENTS ==========


def parse_speed_figures_for_block(block) -> list[int]:
    """Extract speed figures using E2/LP '/' marker, then skip call changes to find fig.

    OPTIMIZED Feb 9 2026: Algorithmic approach handles ANY number of call positions
    (2 for sprints, 3-4 for routes) instead of regex with fixed capture groups.
    """
    figs = []
    if not block:
        logger.debug("parse_speed_figures_for_block: empty block")
        return figs

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for line in block_str.split("\n"):
        parts = line.split()
        # Find the E2/LP marker: a field containing "/"
        slash_idx = None
        for idx, part in enumerate(parts):
            if "/" in part and idx >= 1:
                # Validate: fields around "/" should be numeric (E1 E2/ LP)
                left = part.replace("/", "")
                if left.isdigit() and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    slash_idx = idx
                    break
        if slash_idx is None:
            continue
        # LP is right after the "/" field
        lp_idx = slash_idx + 1
        # After LP, walk forward past call changes until we hit the speed figure
        j = lp_idx + 1
        while j < len(parts):
            raw = parts[j]
            cleaned = raw.lstrip("+-")
            # Strip Unicode superscripts/special chars that BRISNET uses for margins
            cleaned = re.sub(r"[^\d]", "", cleaned)
            if not cleaned:
                break
            try:
                val = int(cleaned)
            except ValueError:
                break
            has_sign = raw[0] in "+-" if raw else False
            # Call changes are small (typically 0-25) and often signed
            # Speed figures are large (40-130) and unsigned
            if has_sign and val < 30:
                j += 1
                continue
            if not has_sign and val < 30:
                j += 1
                continue
            # First number >= 40 without sign is likely the speed figure
            if 40 <= val <= 130:
                figs.append(val)
            break

    return figs[:10]



# ===================== Form Cycle & Recency Analysis =====================


def parse_recent_races_detailed(block) -> list[dict]:
    """Extract recent race history using E2/LP slash marker + jockey name boundary.

    OPTIMIZED Feb 9 2026: Uses jockey name boundary to truncate line before
    scanning for finish positions, preventing decimal odds (e.g., 13.31)
    from contaminating finish position extraction.

    BRISNET race line structure after date+track:
      dist surf class E1 E2/ LP [calls] SPD  PP ST 1C 2C STR FIN   JockeyName  Med Odds ...
    Key: FIN is the last number before 2+ spaces followed by jockey name (alpha chars).
    """
    races = []
    if not block:
        logger.debug("parse_recent_races_detailed: empty block")
        return races

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    today = datetime.now()
    date_pattern = re.compile(r"(\d{2}[A-Za-z]{3}\d{2})[A-Za-z]{2,4}\s+(.+)")

    for m in date_pattern.finditer(block_str):
        date_str = m.group(1)
        rest = m.group(2)
        try:
            race_date = datetime.strptime(date_str, "%d%b%y")
            days_ago = (today - race_date).days

            # Strategy: find the "/" in E2/LP, then after call changes and speed fig,
            # take exactly 6 numbers: PP ST 1C 2C STR FIN
            # But first, truncate line at jockey name to avoid odds contamination
            # Jockey name: 2+ spaces followed by capital letter + lowercase (e.g. "  OrtizIJ")
            jockey_match = re.search(r"(\d)\s{2,}([A-Z][a-zA-Z])", rest)
            if jockey_match:
                rest_trimmed = rest[: jockey_match.start(2)]
            else:
                rest_trimmed = rest

            # Find the "/" separator (E2/LP marker)
            slash_pos = rest_trimmed.find("/")
            if slash_pos < 0:
                continue

            # Extract everything after the "/" section
            after_slash = rest_trimmed[slash_pos + 1 :]
            # Parse all numeric fields from after LP onward
            nums = re.findall(r"[+-]?\d+", after_slash)
            # Format: LP CALL1 CALL2 [CALL3] SPD PP ST 1C 2C STR FIN
            # LP is first, then calls (small, often signed), then SPD (large unsigned),
            # then PP ST 1C 2C STR FIN (all small)
            # Find the speed figure (first number >= 40 after calls)
            spd_idx = None
            for i, s in enumerate(nums):
                val = int(s.lstrip("+-"))
                if i == 0:
                    continue  # Skip LP
                if val >= 40:
                    spd_idx = i
                    break
            if spd_idx is not None and spd_idx + 6 < len(nums):
                # After SPD: PP, ST, 1C, 2C, STR, FIN
                finish = int(nums[spd_idx + 6].lstrip("+-"))
            else:
                # Fallback: take the 6th number counting back from end of trimmed nums
                # (less reliable but better than nothing)
                finish = 99

            races.append({"date": race_date, "days_ago": days_ago, "finish": finish})
        except Exception:
            pass

    return sorted(races, key=lambda x: x["days_ago"])[:6]



def parse_workout_data(block) -> dict:
    """
    ENHANCED: Extract workout information with pattern analysis.
    Returns dict with best_time, num_works, recency, pattern_bonus
    """
    workouts = {
        "best_time": None,
        "num_recent": 0,
        "days_since_last": 999,
        "pattern_bonus": 0.0,
    }

    if not block:
        logger.debug("parse_workout_data: empty block")
        return workouts

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    # Enhanced pattern: captures bullet (×), distance, time, grade, rank
    pattern = r"([×]?)(\d{1,2}[A-Z][a-z]{2})\s+\w+\s+(\d+)f\s+\w+\s+([\d:.«©ª¬®¯°¨]+)\s+([HBG]g?)(?:\s+(\d+)/(\d+))?"

    work_details = []
    for match in re.finditer(pattern, block_str):
        try:
            bullet = match.group(1) == "×"
            distance = int(match.group(3))
            time_str = match.group(4)
            grade = match.group(5)
            rank = int(match.group(6)) if match.group(6) else None
            total = int(match.group(7)) if match.group(7) else None

            time_clean = re.sub(r"[«©ª¬®¯°¨]", "", time_str)
            # CRITICAL FIX: Wrap float conversions in try-except to prevent ValueError
            try:
                if ":" in time_clean:
                    parts = time_clean.split(":")
                    if len(parts) == 2:
                        time_seconds = float(parts[0]) * 60 + float(parts[1])
                    elif len(parts) > 0:
                        time_seconds = float(parts[-1])  # Use last part if malformed
                    else:
                        continue  # Skip this workout if no valid parts
                else:
                    time_seconds = float(time_clean)
            except (ValueError, IndexError):
                continue  # Skip workout if conversion fails

            # CRITICAL FIX: Validate distance > 0 before division
            normalized_time = time_seconds * (4.0 / distance) if distance > 0 else 999

            work_details.append(
                {
                    "bullet": bullet,
                    "distance": distance,
                    "time": normalized_time,
                    "grade": grade,
                    "rank": rank,
                    "total": total,
                }
            )
        except Exception:
            pass

    workouts["num_recent"] = len(work_details)

    if work_details:
        workouts["best_time"] = min(w["time"] for w in work_details)

        # SAVANT: Workout pattern analysis
        bonus = 0.0
        if len(work_details) >= 3:
            times = [w["time"] for w in work_details[:3]]
            if times[0] < times[1] < times[2]:
                bonus += 0.08  # Sharp pattern
            elif times[0] > times[1] > times[2]:
                bonus -= 0.06  # Dull pattern

        if work_details[0]["bullet"]:
            bonus += 0.03  # Recent bullet

        if "g" in work_details[0]["grade"].lower():
            bonus += 0.03  # Gate work

        # CRITICAL FIX: Validate total > 0 before division to prevent ZeroDivisionError
        # NOTE: Single percentile check (was double-counted before Feb 9 fix)
        if (
            work_details[0]["rank"]
            and work_details[0]["total"]
            and work_details[0]["total"] > 0
        ):
            percentile = work_details[0]["rank"] / work_details[0]["total"]
            if percentile <= 0.20:
                bonus += 0.07  # Top 20% of field (elite work)
            elif percentile <= 0.25:
                bonus += 0.04  # Top 25%
            elif percentile <= 0.50:
                bonus += 0.02  # Top half

        bullet_count = sum(1 for w in work_details[:5] if w["bullet"])
        if bullet_count >= 2:
            bonus += 0.05  # Consistent quality

        workouts["pattern_bonus"] = bonus

    return workouts



# ===================== Class Rating Calculator (Comprehensive) =====================


def extract_race_metadata_from_pp_text(pp_text: str) -> dict[str, Any]:
    """
    UNIVERSAL: Extract race type and purse from BRISNET PP text headers.
    Works across ALL tracks, ALL purse levels, ALL race types.

    Returns dict with:
    - purse_amount: int (extracted purse)
    - race_type: str (extracted race type)
    - race_type_clean: str (normalized: 'claiming', 'allowance', 'stakes', etc.)
    - confidence: float (0.0-1.0)
    - detection_method: str (how it was detected)

    Examples of BRISNET headers:
    - "PURSE $6,250. Claiming. For Three Year Olds..."
    - "Purse: $50,000 Allowance Optional Claiming"
    - "$100,000 Grade 2 Stakes"
    - "Race 7: Clm6250n3L"
    """
    if not pp_text or len(pp_text.strip()) < 50:
        return {
            "purse_amount": 0,
            "race_type": "",
            "race_type_clean": "unknown",
            "confidence": 0.0,
            "detection_method": "no_text",
        }

    result = {
        "purse_amount": 0,
        "race_type": "",
        "race_type_clean": "unknown",
        "confidence": 0.0,
        "detection_method": "none",
    }

    # Get first 800 characters (header region)
    header = pp_text[:800]

    # ========== PURSE EXTRACTION (Multi-Pattern) ==========
    purse_patterns = [
        r"PURSE\s+\$([\d,]+)",  # "PURSE $6,250"
        r"Purse:\s+\$([\d,]+)",  # "Purse: $50,000"
        r"\$([\d,]+)\s+(?:Grade|Stakes|Allowance|Claiming)",  # "$100,000 Grade 2"
        r"\$([\d,]+)",  # Any dollar amount in header
    ]

    for pattern in purse_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            try:
                result["purse_amount"] = int(match.group(1).replace(",", ""))
                result["detection_method"] = "purse_text"
                result["confidence"] = 0.9
                break
            except Exception:                pass

    # ========== RACE TYPE EXTRACTION (Multi-Pattern) ==========
    race_type_patterns = [
        # Graded Stakes
        (r"Grade\s+(I{1,3}|[123])", "stakes_graded", 1.0),
        (r"G([123])", "stakes_graded", 1.0),
        # Stakes
        (r"Stakes", "stakes", 0.95),
        (r"Handicap", "stakes", 0.9),
        (r"Listed", "stakes", 0.9),
        # Allowance
        (r"Allowance\s+Optional\s+Claiming", "allowance_optional", 0.95),
        (r"Optional\s+Claiming", "allowance_optional", 0.95),
        (r"Allowance", "allowance", 0.95),
        (r"\bAOC\b", "allowance_optional", 0.9),
        (r"\bAlw\b", "allowance", 0.9),
        # Claiming
        (r"Maiden\s+Claiming", "maiden_claiming", 0.95),
        (r"Maiden\s+Clm", "maiden_claiming", 0.9),
        (r"\bMCL\b", "maiden_claiming", 0.9),
        (r"Claiming", "claiming", 0.95),
        (r"\bClm\b", "claiming", 0.9),
        (r"\bMC\b", "claiming", 0.85),
        # Maiden
        (r"Maiden\s+Special\s+Weight", "maiden_special_weight", 0.95),
        (r"\bMSW\b", "maiden_special_weight", 0.9),
        (r"Maiden", "maiden", 0.85),
        # Starter
        (r"Starter\s+Optional\s+Claiming", "starter_optional_claiming", 0.9),
        (r"Starter\s+Allowance", "starter_allowance", 0.9),
        (r"Starter\s+Handicap", "starter_handicap", 0.9),
        (r"\bSOC\b", "starter_optional_claiming", 0.85),
        # Waiver
        (r"Waiver", "waiver_claiming", 0.85),
    ]

    for pattern, race_type_clean, confidence in race_type_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            result["race_type"] = match.group(0)
            result["race_type_clean"] = race_type_clean
            result["confidence"] = max(result["confidence"], confidence)
            if result["detection_method"] == "none":
                result["detection_method"] = "text_pattern"
            break

    # ========== EMBEDDED RACE TYPE (Clm25000n2L format) ==========
    embedded_pattern = r"\b(Clm|MC|OC|Alw|Mdn|MSW|Stk|G[123])([\d]+[kK]?)"
    embedded_match = re.search(embedded_pattern, header, re.IGNORECASE)
    if embedded_match:
        prefix = embedded_match.group(1).lower()
        amount_str = embedded_match.group(2)

        # Extract purse from embedded format if not already found
        if result["purse_amount"] == 0:
            try:
                if "k" in amount_str.lower():
                    result["purse_amount"] = int(amount_str[:-1]) * 1000
                else:
                    result["purse_amount"] = int(amount_str)
                result["detection_method"] = "embedded_format"
            except Exception:                pass

        # Map prefix to race type if not already found
        if result["race_type_clean"] == "unknown":
            prefix_map = {
                "clm": "claiming",
                "mc": "claiming",
                "oc": "allowance_optional",
                "alw": "allowance",
                "mdn": "maiden",
                "msw": "maiden_special_weight",
                "stk": "stakes",
                "g1": "stakes_graded",
                "g2": "stakes_graded",
                "g3": "stakes_graded",
            }
            result["race_type_clean"] = prefix_map.get(prefix, "unknown")
            result["race_type"] = embedded_match.group(0)
            result["confidence"] = 0.85
            if result["detection_method"] == "none":
                result["detection_method"] = "embedded_format"

    return result



def infer_purse_from_race_type(race_type: str) -> int | None:
    """
    LEGACY: Infer purse from race type names like 'Clm25000n2L' or 'MC50000'.
    BRISNET embeds purse values in race type strings.

    NOTE: Use extract_race_metadata_from_pp_text() for comprehensive detection.
    This function is kept for backward compatibility.

    Examples:
    - 'Clm25000n2L' → $25,000
    - 'MC50000' → $50,000
    - 'OC20k' → $20,000
    - 'Alw28000' → $28,000
    """
    if not race_type:
        return None

    # Pattern 1: Direct numbers (Clm25000, MC50000, Alw28000)
    match = re.search(r"(\d{4,6})", race_type)
    if match:
        return int(match.group(1))

    # Pattern 2: With 'k' suffix (OC20k, Alw50k)
    match = re.search(r"(\d+)k", race_type, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000

    # Pattern 3: Common defaults by type
    race_lower = race_type.lower()
    if "maiden" in race_lower or "mdn" in race_lower or "md sp wt" in race_lower:
        return 50000  # Typical maiden special weight
    elif "claiming" in race_lower or "clm" in race_lower or "mc" in race_lower:
        return 25000  # Typical claiming level
    elif "allowance" in race_lower or "alw" in race_lower:
        return 50000  # Typical allowance
    elif (
        "stake" in race_lower
        or "stk" in race_lower
        or "g1" in race_lower
        or "g2" in race_lower
        or "g3" in race_lower
    ):
        return 100000  # Stakes minimum

    return None



def parse_recent_class_levels(block) -> list[dict]:
    """
    Parse recent races to extract class progression data.
    CRITICAL FIX: Infers purses from race type names since BRISNET embeds them.
    Returns list of dicts with purse, race_type, finish_position
    """
    races = []
    if not block:
        logger.debug("parse_recent_class_levels: empty block")
        return races
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    # Enhanced pattern to match BRISNET format
    # Example: "11Jan26SAª 6½ ft :21ª :44¨1:09« 1:16© ¡ ¨¨¨ Clm25000n2L ¨¨©"
    lines = block_str.split("\n")
    for line in lines:
        # Updated pattern to capture embedded race types
        race_match = re.search(
            r"(\d{2}[A-Za-z]{3}\d{2})\w+\s+[\d½]+[f]?\s+.*?"
            r"([A-Z][a-z]{2,}\d+[a-zA-Z0-9\-]*|MC\d+|OC\d+k?|Alw\d+|Stk|G[123]|Hcp)",
            line,
        )

        if race_match:
            race_type = race_match.group(2)

            # CRITICAL: Infer purse from race type name
            inferred_purse = infer_purse_from_race_type(race_type)

            # Extract finish position from same line (look for FIN column)
            # Pattern: FIN followed by position like "1st", "2nd", "3©", "4¬", etc.
            finish_match = re.search(r"FIN\s+(\d{1,2})[ƒ®«ª³©¨°¬²‚±\s]", line)
            finish_pos = int(finish_match.group(1)) if finish_match else 0

            with contextlib.suppress(Exception):
                races.append(
                    {
                        "race_type": race_type,
                        "purse": inferred_purse if inferred_purse else 0,
                        "finish_pos": finish_pos,
                    }
                )

    return races[:5]  # Last 5 races


# Purse


def detect_purse_amount(pp_text: str) -> int | None:
    s = pp_text or ""
    m = re.search(r"(?mi)\bPurse\b[^$\n\r]*\$\s*([\d,]+)", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:            pass
    m = re.search(r"(?mi)\b(?:Added|Value)\b[^$\n\r]*\$\s*([\d,]+)", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:            pass
    m = re.search(
        r"(?mi)\b(Mdn|Maiden|Allowance|Alw|Claiming|Clm|Starter|Stake|Stakes)\b[^:\n\r]{0,50}\b(\d{2,4})\s*[Kk]\b",
        s,
    )
    if m:
        try:
            return int(m.group(2)) * 1000
        except Exception:            pass
    m = re.search(r"(?m)\$\s*([\d,]{5,})", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:            pass
    return None



# ======================== Phase 1: Enhanced Parsing Functions ========================


def parse_pace_speed_pars(pp_text: str) -> dict[str, int]:
    """Parse BRIS Pace & Speed Pars from the PP header area.

    These are the race-level E1, E2/Late, and SPD pars — the AVERAGE pace/speed
    ratings for the LEADER/WINNER at today's class level and distance.
    Comparing a horse's figures to these pars reveals if they're above or below
    class standard.

    Format in PP text: "E1  E2/LATE  SPD" followed by "86  88/ 87  88"
    Or inline: "E1 E2/Late Spd" headers with values on next line.

    Returns: {"e1_par": int, "e2_par": int, "lp_par": int, "spd_par": int}
    """
    pars: dict[str, int] = {}
    if not pp_text:
        logger.debug("parse_pace_speed_pars: empty pp_text")
        return pars

    # Pattern 1: "E1  E2/LATE  SPD" on one line, values on next
    # or "E1  E2/ LATE  SPD" format, then "86  88/ 87  88"
    header_match = re.search(
        r"E1\s+E2\s*/?\s*(?:LATE|Late)\s+SPD\s*\n\s*(\d{2,3})\s+(\d{2,3})\s*/?\s*(\d{2,3})\s+(\d{2,3})",
        pp_text,
        re.IGNORECASE,
    )
    if header_match:
        pars["e1_par"] = int(header_match.group(1))
        pars["e2_par"] = int(header_match.group(2))
        pars["lp_par"] = int(header_match.group(3))
        pars["spd_par"] = int(header_match.group(4))
        return pars

    # Pattern 2: Inline "E1 86 E2/Late 88/87 SPD 88" or similar
    inline_match = re.search(
        r"E1\s+(\d{2,3})\s+E2\s*/?\s*(?:Late)?\s*(\d{2,3})\s*/?\s*(\d{2,3})\s+SPD\s+(\d{2,3})",
        pp_text,
        re.IGNORECASE,
    )
    if inline_match:
        pars["e1_par"] = int(inline_match.group(1))
        pars["e2_par"] = int(inline_match.group(2))
        pars["lp_par"] = int(inline_match.group(3))
        pars["spd_par"] = int(inline_match.group(4))
        return pars

    # Pattern 3: "BRIS Pace & Speed Pars" section with separated values
    pars_match = re.search(
        r"(?:Pace|PACE).*?(?:Speed|SPD).*?Pars?\s*\]?.*?(\d{2,3})\s+(\d{2,3})\s*/?\s*(\d{2,3})\s+(\d{2,3})",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if pars_match:
        pars["e1_par"] = int(pars_match.group(1))
        pars["e2_par"] = int(pars_match.group(2))
        pars["lp_par"] = int(pars_match.group(3))
        pars["spd_par"] = int(pars_match.group(4))

    if not pars:
        logger.debug(
            "parse_pace_speed_pars: no par pattern matched in %d chars", len(pp_text)
        )
    return pars



def parse_quickplay_comments(horse_block: str) -> dict[str, list[str]]:
    """Parse QuickPlay positive (★) and negative (●) comments for a horse.

    These Brisnet-generated comments are pre-analyzed signals appearing at the
    top of each horse's PP section:
    - ★ (star) = Positive comment (e.g., "Won last race", "Rail post winning at 19%")
    - ● (bullet) = Negative comment (e.g., "Moves up in class", "Poor Speed Figures")

    Returns: {"positive": ["Won last race", ...], "negative": ["Poor Speed Figures", ...]}
    """
    result: dict[str, list[str]] = {"positive": [], "negative": []}
    if not horse_block:
        logger.debug("parse_quickplay_comments: empty horse_block")
        return result

    # Find the comments section — appears between header stats and race lines
    # Look for lines starting with ★ or ● (or text equivalents)
    for line in horse_block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Positive signals (★ or star marker)
        star_comments = re.findall(r"[★\u2605]\s*([^★●\u2605\u25CF]+)", line)
        for c in star_comments:
            c = c.strip()
            if len(c) > 5:  # Filter noise
                result["positive"].append(c)

        # Negative signals (● or bullet marker)
        bullet_comments = re.findall(r"[●\u25CF]\s*([^★●\u2605\u25CF]+)", line)
        for c in bullet_comments:
            c = c.strip()
            if len(c) > 5:
                result["negative"].append(c)

    return result



def parse_bris_rr_cr_per_race(horse_block: str) -> list[dict[str, int]]:
    """Parse BRIS Race Rating (RR) and Class Rating (CR) from past race lines.

    In the PP, each past race line has RR and CR in columns before the race type.
    Format: "DATE TRACK  DIST  FRACS  AGE  RR CR RACETYPE"
    Example: "03Jan26CT  1 1/16 ft  :24  :49  1:15  34  106  Mdn 32k"
    The RR appears before CR, both appear before the RACETYPE column.

    Pattern from PP: "34 112 79 75/"  where first number after age is RR~CR combined
    Actually in BRISNET format: position between fractional times and race type

    Returns: [{"rr": 106, "cr": 112}, ...] most recent first
    """
    races = []
    if not horse_block:
        logger.debug("parse_bris_rr_cr_per_race: empty horse_block")
        return races

    # Match race lines by date pattern
    # Format: DDMonYYTrack  distance  fracs  RR CR RACETYPE
    # The RR and CR appear as two numbers right before the RACETYPE keyword
    for line in horse_block.split("\n"):
        # Only process race lines (start with date like "03Jan26")
        date_match = re.match(r"(\d{2}[A-Za-z]{3}\d{2})", line.strip())
        if not date_match:
            continue

        # Find RR CR before racetype keywords
        # Pattern: ... number  number  Mdn|Clm|Alw|OC|MC|Stk|G1|G2|G3|Hcp
        rr_cr_match = re.search(
            r"(\d{2,3})\s+(\d{2,3})\s+(?:Mdn|Clm|Alw|OC|MC|Stk|Hcp|©|G1|G2|G3)",
            line,
        )
        if rr_cr_match:
            try:
                rr = int(rr_cr_match.group(1))
                cr = int(rr_cr_match.group(2))
                # Sanity check: RR and CR should be in reasonable range (50-120)
                if 30 <= rr <= 130 and 30 <= cr <= 130:
                    races.append({"rr": rr, "cr": cr})
            except (ValueError, TypeError):
                pass

    return races[:6]  # Last 6 races



def parse_track_bias_stats(pp_text: str) -> dict[str, any]:
    """Parse comprehensive Track Bias Stats from BRISNET PP text.

    Extracts fields that Impact Values alone don't capture:
    - pct_wire: % of races won wire-to-wire (high = E dominant)
    - speed_bias_pct: % races won by E or E/P styles (>60% = speed favoring)
    - wnr_avg_bl_1st: Winner's avg beaten lengths at 1st call (low = pace contested)
    - wnr_avg_bl_2nd: Winner's avg beaten lengths at 2nd call
    - pct_races_won: {E: x%, E/P: x%, P: x%, S: x%} win distribution
    - num_races: Sample size for confidence weighting

    Prefers Week Totals over Meet Totals when both available.
    """
    stats: dict[str, any] = {}
    if not pp_text:
        logger.debug("parse_track_bias_stats: empty pp_text")
        return stats

    # --- %Wire ---
    # Format: "%Wire: 63%" or "%Wire:  77%"
    wire_match = re.search(r"%Wire:\s*(\d+)%", pp_text, re.IGNORECASE)
    if wire_match:
        stats["pct_wire"] = int(wire_match.group(1))

    # --- Speed Bias ---
    # Format: "Speed Bias: 89%" or "Speed Bias: 92%"
    # Prefer Week Totals (search after 'Week Totals' first, then Meet)
    for section_marker in [r"\*\s*Week\s+Totals\s*\*", r"\*\s*MEET\s+Totals\s*\*"]:
        section_match = re.search(
            section_marker + r".*?Speed Bias:\s*(\d+)%",
            pp_text,
            re.DOTALL | re.IGNORECASE,
        )
        if section_match:
            stats["speed_bias_pct"] = int(section_match.group(1))
            break

    # --- WnrAvgBL (Winner's Average Beaten Lengths) ---
    # Format: "WnrAvgBL" header, then "1stCall: 0.8" and "2ndCall: 0.4"
    # Prefer Week Totals
    for section_marker in [r"\*\s*Week\s+Totals\s*\*", r"\*\s*MEET\s+Totals\s*\*"]:
        wnr_section = re.search(
            section_marker + r".*?WnrAvgBL.*?1stCall:\s*([\d.]+).*?2ndCall:\s*([\d.]+)",
            pp_text,
            re.DOTALL | re.IGNORECASE,
        )
        if wnr_section:
            stats["wnr_avg_bl_1st"] = float(wnr_section.group(1))
            stats["wnr_avg_bl_2nd"] = float(wnr_section.group(2))
            break

    # --- % Races Won per runstyle ---
    # Format after Runstyle Impact Values row:
    # "%Races Won   69%   20%   6%   5%" (E, E/P, P, S)
    for section_marker in [r"\*\s*Week\s+Totals\s*\*", r"\*\s*MEET\s+Totals\s*\*"]:
        pct_match = re.search(
            section_marker + r".*?%Races Won\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%",
            pp_text,
            re.DOTALL | re.IGNORECASE,
        )
        if pct_match:
            stats["pct_races_won"] = {
                "E": int(pct_match.group(1)),
                "E/P": int(pct_match.group(2)),
                "P": int(pct_match.group(3)),
                "S": int(pct_match.group(4)),
            }
            break

    # --- Post Bias Avg Win % ---
    # Format: "Avg Win %  17%  14%  13%  7%" (RAIL, 1-3, 4-7, 8+)
    for section_marker in [r"\*\s*Week\s+Totals\s*\*", r"\*\s*MEET\s+Totals\s*\*"]:
        post_win_match = re.search(
            section_marker + r".*?Avg Win\s*%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%",
            pp_text,
            re.DOTALL | re.IGNORECASE,
        )
        if post_win_match:
            stats["post_avg_win_pct"] = {
                "rail": int(post_win_match.group(1)),
                "inner": int(post_win_match.group(2)),
                "mid": int(post_win_match.group(3)),
                "outer": int(post_win_match.group(4)),
            }
            break

    # --- # Races (sample size) ---
    # Format: "# Races: 65" or "# Races: 13"
    races_match = re.search(r"# Races:\s*(\d+)", pp_text, re.IGNORECASE)
    if races_match:
        stats["num_races"] = int(races_match.group(1))

    return stats



def parse_race_summary_rankings(pp_text: str) -> dict[str, dict[str, float]]:
    """Parse Race Summary ranking tables from BRISNET PP text.

    The Race Summary section contains multiple ranking categories:
    - Speed Last Race: Last speed fig per horse
    - Back Speed: Best speed at today's dist/surf within 1 year
    - Current Class: Class rating emphasizing recent at today's dist/surf
    - Average Class Last 3: Avg class of last 3 starts
    - Prime Power: Combined handicapping rating
    - Early Pace Last Race: E1/E2 pace from last race
    - Late Pace Last Race: Late pace from last race

    Returns dict like:
    {
        "Speed Last Race": {"Cherokee Castle": 72, "Cowgirl Attitude": 74, ...},
        "Current Class": {"Cherokee Castle": 109.4, ...},
        ...
    }
    """
    rankings: dict[str, dict[str, float]] = {}

    # Each ranking section in the Race Summary follows this layout:
    # HEADER LINE (bold category name)
    # number horsename value
    # number horsename value
    # ... (one per horse)

    # Categories to parse — these are the column headers in the Race Summary
    category_patterns = [
        ("Speed Last Race", r"Speed\s+Last\s+Race"),
        ("Back Speed", r"Back\s+Speed"),
        ("Current Class", r"Current\s+Class"),
        ("Average Class Last 3", r"Average\s+Class\s+Last\s+3"),
        ("Prime Power", r"Prime\s+Power"),
        ("Early Pace Last Race", r"Early\s+Pace\s*(?:Last\s+Race)?"),
        ("Late Pace Last Race", r"Late\s+Pace\s*(?:Last\s+Race)?"),
    ]

    for cat_name, pattern in category_patterns:
        # Find section header
        section_match = re.search(pattern, pp_text, re.IGNORECASE)
        if not section_match:
            continue

        # Extract lines after the header — ranking entries
        start_pos = section_match.end()
        remaining = pp_text[start_pos : start_pos + 600]  # Enough for ~12 horses
        lines = remaining.strip().split("\n")

        cat_rankings: dict[str, float] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match: "74 Cowgirl Attitude" or "109.4 Cherokee Castle"
            # Format is: value horsename (rankings are sorted by value)
            entry_match = re.match(r"^([\d.]+)\s+([A-Za-z][A-Za-z\s']+?)\s*$", line)
            if entry_match:
                value = float(entry_match.group(1))
                horse = entry_match.group(2).strip()
                cat_rankings[horse] = value
            else:
                # Also try: "number horsename value" format used in some sections
                entry_match2 = re.match(
                    r"^(\d+)\s+([A-Za-z][A-Za-z\s']+?)\s+([\d.]+)\s*$", line
                )
                if entry_match2:
                    value = float(entry_match2.group(3))
                    horse = entry_match2.group(2).strip()
                    cat_rankings[horse] = value
                elif cat_rankings:
                    # Non-matching line after we found entries = end of section
                    break

        if cat_rankings:
            rankings[cat_name] = cat_rankings

    return rankings



def parse_track_bias_impact_values(pp_text: str) -> dict[str, float]:
    """Extract Track Bias Impact Values from '9b. Track Bias (Numerical)' section
    OR from raw BRISNET 'Week Totals' / 'MEET Totals' format."""
    impact_values = {}

    # METHOD 1: Formatted '9b. Track Bias' section
    bias_match = re.search(
        r"9b\.\s*Track Bias.*?\n(.*?)(?=\n\d+[a-z]?\.|$)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if bias_match:
        bias_text = bias_match.group(1)
        for match in re.finditer(
            r"-\s*([A-Z/]+)\s*\([^)]+\):\s*Impact Value\s*=\s*([\d.]+)", bias_text
        ):
            style_code = match.group(1).strip()
            impact_val = float(match.group(2))
            impact_values[style_code] = impact_val
        if impact_values:
            return impact_values

    # METHOD 2: Raw BRISNET 'Week Totals' format (preferred — most recent)
    # Runstyle: E E/P P S
    # ...
    # Impact Values: 1.17 1.14 1.18 0.32
    week_match = re.search(
        r"\*\s*Week\s+Totals\s*\*.*?"
        r"Runstyle:\s*E\s+E/P\s+P\s+S.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if week_match:
        impact_values["E"] = float(week_match.group(1))
        impact_values["E/P"] = float(week_match.group(2))
        impact_values["P"] = float(week_match.group(3))
        impact_values["S"] = float(week_match.group(4))
        return impact_values

    # METHOD 3: Meet Totals fallback
    meet_match = re.search(
        r"\*\s*MEET\s+Totals\s*\*.*?"
        r"Runstyle:\s*E\s+E/P\s+P\s+S.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if meet_match:
        impact_values["E"] = float(meet_match.group(1))
        impact_values["E/P"] = float(meet_match.group(2))
        impact_values["P"] = float(meet_match.group(3))
        impact_values["S"] = float(meet_match.group(4))

    return impact_values


# ======================== RACE AUDIT ENHANCEMENTS (Feb 2026) ========================
# Oaklawn R1 Audit: Model ranked Tell Me When last (#7) but horse WON.
# Model's #1 Tiffany Twist finished 4th. Root cause: 5 gaps in bias/angle handling.
# All enhancements below are ADDITIVE — no existing logic is modified.


def parse_weekly_post_bias(pp_text: str) -> dict[str, float]:
    """
    Extract weekly post-position Impact Values from BRISNET PP '9b' section
    OR from raw BRISNET Track Bias Stats format.

    Returns dict mapping individual post numbers (as strings) to their impact values.
    """
    post_impacts: dict[str, float] = {}

    # METHOD 1: Formatted '9b. Track Bias' section
    bias_match = re.search(
        r"9b\.\s*Track Bias.*?\n(.*?)(?=\n\d+[a-z]?\.|$)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if bias_match:
        bias_text = bias_match.group(1)
        for match in re.finditer(
            r"-\s*Posts?\s*(\d+)[-–](\d+)\s*\([^)]*\):\s*Impact Value\s*=\s*([\d.]+)",
            bias_text,
        ):
            low_post = int(match.group(1))
            high_post = int(match.group(2))
            impact_val = float(match.group(3))
            for p in range(low_post, high_post + 1):
                post_impacts[str(p)] = impact_val
        for match in re.finditer(
            r"-\s*Post\s*(\d+)\s*\([^)]*\):\s*Impact Value\s*=\s*([\d.]+)", bias_text
        ):
            post_impacts[match.group(1)] = float(match.group(2))
        if post_impacts:
            return post_impacts

    # METHOD 2: Raw BRISNET 'Week Totals' format (preferred — most recent)
    # Post Bias: RAIL 1-3 4-7 8+
    # ...
    # Impact Values: 2.59 1.72 0.68 0.70
    week_post_match = re.search(
        r"\*\s*Week\s+Totals\s*\*.*?"
        r"Post Bias:\s*RAIL\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s+(\d+)\+.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if week_post_match:
        inner_low = int(week_post_match.group(1))
        inner_high = int(week_post_match.group(2))
        mid_low = int(week_post_match.group(3))
        mid_high = int(week_post_match.group(4))
        outer_start = int(week_post_match.group(5))
        rail_impact = float(week_post_match.group(6))
        inner_impact = float(week_post_match.group(7))
        mid_impact = float(week_post_match.group(8))
        outer_impact = float(week_post_match.group(9))
        # RAIL = post 1 (separate from the inner range)
        post_impacts["1"] = rail_impact
        for p in range(inner_low, inner_high + 1):
            if str(p) not in post_impacts:
                post_impacts[str(p)] = inner_impact
        for p in range(mid_low, mid_high + 1):
            post_impacts[str(p)] = mid_impact
        for p in range(outer_start, outer_start + 6):
            post_impacts[str(p)] = outer_impact
        return post_impacts

    # METHOD 3: Meet Totals fallback
    meet_post_match = re.search(
        r"\*\s*MEET\s+Totals\s*\*.*?"
        r"Post Bias:\s*RAIL\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s+(\d+)\+.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if meet_post_match:
        inner_low = int(meet_post_match.group(1))
        inner_high = int(meet_post_match.group(2))
        mid_low = int(meet_post_match.group(3))
        mid_high = int(meet_post_match.group(4))
        outer_start = int(meet_post_match.group(5))
        rail_impact = float(meet_post_match.group(6))
        inner_impact = float(meet_post_match.group(7))
        mid_impact = float(meet_post_match.group(8))
        outer_impact = float(meet_post_match.group(9))
        post_impacts["1"] = rail_impact
        for p in range(inner_low, inner_high + 1):
            if str(p) not in post_impacts:
                post_impacts[str(p)] = inner_impact
        for p in range(mid_low, mid_high + 1):
            post_impacts[str(p)] = mid_impact
        for p in range(outer_start, outer_start + 6):
            post_impacts[str(p)] = outer_impact

    return post_impacts



def parse_pedigree_spi(pp_text: str) -> dict[str, int | None]:
    """Extract SPI (Sire Performance Index) from pedigree sections"""
    spi_values = {}

    # Look for pattern like "Sire: Hard Spun (SPI: 1.30)" in Section 4
    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for SPI in Sire line
        spi_match = re.search(r"Sire:.*?SPI:\s*([\d.]+)", section, re.IGNORECASE)
        if spi_match:
            try:
                spi = int(float(spi_match.group(1)) * 100)  # Convert 1.30 to 130
                spi_values[horse_name] = spi
            except Exception:
                spi_values[horse_name] = None
        else:
            spi_values[horse_name] = None

    return spi_values



def parse_pedigree_surface_stats(pp_text: str) -> dict[str, dict[str, any]]:
    """Extract surface statistics (Turf/AW win%) from pedigree sections"""
    surface_stats = {}

    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        stats = {}
        # Look for "Turf: 12% (class-adj)" or similar
        turf_match = re.search(r"Turf:\s*([\d.]+)%", section, re.IGNORECASE)
        if turf_match:
            stats["turf_pct"] = float(turf_match.group(1))

        # Look for "AW: 8% (class-adj)" or similar
        aw_match = re.search(r"(?:AW|All-Weather):\s*([\d.]+)%", section, re.IGNORECASE)
        if aw_match:
            stats["aw_pct"] = float(aw_match.group(1))

        if stats:
            surface_stats[horse_name] = stats

    return surface_stats



def parse_awd_analysis(pp_text: str) -> dict[str, str]:
    """Extract AWD (Avg Winning Distance) analysis from pedigree sections"""
    awd_data = {}

    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for "✔ AWD Match" or "⚠ Distance Mismatch"
        if "✔ AWD Match" in section or "AWD Match" in section:
            awd_data[horse_name] = "match"
        elif "⚠ Distance Mismatch" in section or "Distance Mismatch" in section:
            awd_data[horse_name] = "mismatch"

    return awd_data



def parse_jockey_combo_stats(section: str) -> tuple[float, float]:
    """
    Parse jockey stats and combo percentage from BRISNET format.

    Searches for patterns like:
    - "Jky: LastName FirstName (starts wins-places-shows win%)"
    - Combo patterns: "w/TrnrLastName: 50 22% 18% 40%" (starts, win%, place%, combo%)

    Returns:
        tuple: (jockey_win_rate, combo_win_rate)
    """
    jockey_win_rate = 0.0
    combo_win_rate = 0.0

    # JOCKEY: Search for "Jky:" pattern
    jockey_pattern = r"Jky:.*?\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)"
    jockey_match = re.search(jockey_pattern, section)
    if jockey_match:
        j_starts = int(jockey_match.group(1))
        j_win_pct = int(jockey_match.group(5)) / 100.0
        if j_starts >= 20:
            jockey_win_rate = j_win_pct

    # COMBO: Search for trainer/jockey combo patterns
    # Format: "w/TrnrName: 50 22% 18% 40%" or similar
    combo_pattern = r"w/.*?:\s*(\d+)\s+(\d+)%\s+(\d+)%\s+(\d+)%"
    combo_match = re.search(combo_pattern, section)
    if combo_match:
        combo_starts = int(combo_match.group(1))
        combo_pct = (
            int(combo_match.group(4)) / 100.0
        )  # Last percentage is combo win rate
        if combo_starts >= 10:  # Minimum 10 starts for combo stats
            combo_win_rate = combo_pct

    return jockey_win_rate, combo_win_rate



# ============ STRUCTURED RACE HISTORY PARSING (Feb 10, 2026) ============


def parse_race_history_from_block(block: str) -> list[dict]:
    """
    Parse structured race history from a single horse's BRISNET PP block.
    Extracts per-race: date, track, surface, distance (furlongs), race type,
    finish, speed figure.

    Surface codes: ft/fst=Dirt, sy/sly=Dirt, my=Dirt, fm/frm=Turf,
                  yl=Turf, sf=Turf, gf=Turf, hy=Turf, aw/tp=Synthetic
    """
    if not block:
        return []

    races = []
    surface_map = {
        "ft": "Dirt",
        "fst": "Dirt",
        "fast": "Dirt",
        "sy": "Dirt",
        "sly": "Dirt",
        "sloppy": "Dirt",
        "my": "Dirt",
        "mdy": "Dirt",
        "muddy": "Dirt",
        "gd": "Dirt",
        "fm": "Turf",
        "frm": "Turf",
        "firm": "Turf",
        "yl": "Turf",
        "yld": "Turf",
        "yielding": "Turf",
        "sf": "Turf",
        "sft": "Turf",
        "soft": "Turf",
        "gf": "Turf",
        "gd-fm": "Turf",
        "hy": "Turf",
        "aw": "Synthetic",
        "tp": "Synthetic",
        "syn": "Synthetic",
    }

    for line in block.split("\n"):
        # Running lines start with date (ddMMMyy) + track code
        date_match = re.match(r"(\d{2}[A-Za-z]{3}\d{2})(\w{2,4})\d*\s+", line.strip())
        if not date_match:
            continue

        try:
            date_str = date_match.group(1)
            track_code = date_match.group(2)

            # Surface detection
            surface = "Dirt"  # Default
            surface_area = line[:80].lower()
            for code, surf in surface_map.items():
                if re.search(rf"\b{re.escape(code)}\b", surface_area):
                    surface = surf
                    break
            # BRISNET turf indicator ⓘ
            if "\u24d8" in line or "\u2460" in line or "ⓘ" in line:
                surface = "Turf"

            # Distance extraction (furlongs)
            distance_f = 0.0
            dist_match = re.search(r"(\d+)\s*½?\s*f(?:ur)?", line[:60], re.IGNORECASE)
            if dist_match:
                base = int(dist_match.group(1))
                distance_f = base + (0.5 if "½" in line[:60] else 0.0)
            else:
                mile_match = re.search(
                    r"(\d+)\s*(?:(\d+)/(\d+))?\s*m(?:ile)?", line[:60], re.IGNORECASE
                )
                if mile_match:
                    miles = int(mile_match.group(1))
                    if mile_match.group(2) and mile_match.group(3):
                        miles += int(mile_match.group(2)) / int(mile_match.group(3))
                    distance_f = miles * 8.0

            # Race type
            race_type = ""
            type_match = re.search(
                r"(Clm\d+[a-zA-Z0-9/.\-]*|MC\d+[a-zA-Z0-9/.\-]*|"
                r"Mdn\d*|Md\s*Sp\s*Wt|Alw\d*[a-zA-Z0-9/.\-]*|"
                r"OC\d+[a-zA-Z0-9/.\-]*|Stk[a-zA-Z0-9/.\-]*|"
                r"G[123]\s*\w*|Hcp\d*|Moc\d*|S\s*Mdn\s*\d*k?)",
                line,
                re.IGNORECASE,
            )
            if type_match:
                race_type = type_match.group(1).strip()

            # Speed figure (first 2-3 digit number between 20-130 after col 40)
            speed_fig = 0
            for spd_str in re.findall(r"\b(\d{2,3})\b", line[40:]):
                fig = int(spd_str)
                if 20 <= fig <= 130:
                    speed_fig = fig
                    break

            # Finish position: scan position numbers in running-line area
            # (matches elite_parser_v2_gold._parse_race_history logic)
            finish_pos = 0
            fin_matches = re.findall(r"(\d{1,2})[ƒ®«ª³©¨°¬²‚±¹²³⁴⁵⁶⁷⁸⁹⁰ⁿ]*", line[80:])
            if fin_matches:
                for fm in reversed(fin_matches):
                    try:
                        f_val = int(fm)
                        if 1 <= f_val <= 20:
                            finish_pos = f_val
                            break
                    except ValueError:
                        continue

            if date_str and (distance_f > 0 or race_type):
                races.append(
                    {
                        "date": date_str,
                        "track": track_code,
                        "surface": surface,
                        "distance": str(distance_f) + "f" if distance_f > 0 else "",
                        "distance_f": round(distance_f, 1),
                        "race_type": race_type,
                        "speed_fig": speed_fig,
                        "finish_pos": finish_pos,
                    }
                )
        except Exception:
            continue

    return races[:10]

