"""
Test Tampa Bay Downs race header parsing fix
"""

import re


def parse_brisnet_race_header_test(header_line: str) -> dict:
    """
    Lightweight test version of parse_brisnet_race_header with fixes
    """
    result = {}

    text = header_line

    # Strip "Ultimate PP's w/ QuickPlay Comments" prefix
    # FIXED: \s* instead of \s+ to handle "CommentsTampa" (no space)
    text = re.sub(r"^Ultimate PP.*?Comments\s*", "", text, flags=re.IGNORECASE).strip()

    print(f"After stripping prefix: '{text}'")

    # Extract track name (simplified)
    # Try to find track name before race type
    track_pattern = (
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:MC|MSW|CLM|AOC|STK|G\d)"
    )
    track_match = re.search(track_pattern, text)
    if track_match:
        result["track_name"] = track_match.group(1)
        print(f"Track name extracted: '{result['track_name']}'")

    # Extract race type and purse
    type_purse_match = re.search(r"([©¨§]?\w+)\s+(\d+)", text)
    if type_purse_match:
        race_type_abbr = type_purse_match.group(1)
        purse = int(type_purse_match.group(2))
        result["race_type_raw"] = race_type_abbr
        result["purse_amount"] = purse

        # Normalize race type
        if race_type_abbr.upper() == "MC":
            result["race_type_normalized"] = "maiden claiming"
        elif race_type_abbr.upper() == "MSW":
            result["race_type_normalized"] = "maiden special weight"
        elif race_type_abbr.upper() in ["CLM", "CL"]:
            result["race_type_normalized"] = "claiming"
        else:
            result["race_type_normalized"] = race_type_abbr.lower()

        print(f"Race type: {race_type_abbr} → {result['race_type_normalized']}")
        print(f"Purse: ${purse:,}")

    return result


def detect_race_type_test(pp_text: str) -> str:
    """
    Test version with MC detection fix
    """
    s = (pp_text or "")[:1000].lower()

    # Check for standalone "MC" followed by number (FIXED)
    if re.search(r"\bmc\s*\d", s):
        return "maiden claiming"

    # Check for maiden patterns
    if re.search(r"\b(mdn|maiden)\b", s):
        if re.search(r"(mcl|mdn\s*clm|maiden\s*claim)", s):
            return "maiden claiming"
        if re.search(r"(msw|maiden\s*special|maiden\s*sp\s*wt)", s):
            return "maiden special weight"
        return "maiden special weight"

    # Claiming
    if re.search(r"\bclm|claiming\b", s):
        return "claiming"

    return "allowance"


if __name__ == "__main__":
    print("=" * 80)
    print("TAMPA BAY DOWNS PARSING TEST")
    print("=" * 80)
    print()

    # User's actual failing header
    test_header = "Ultimate PP's w/ QuickPlay CommentsTampa Bay Downs MC 16000 1 Mile (T) 3yo Wednesday, February 11, 2026 Race 4"

    print(f"Input header:\n{test_header}")
    print()

    result = parse_brisnet_race_header_test(test_header)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("RACE TYPE DETECTION TEST:")
    print("=" * 80)
    detected_type = detect_race_type_test(test_header)
    print(f"  Detected race type: '{detected_type}'")

    print("\n" + "=" * 80)
    print("VALIDATION:")
    print("=" * 80)

    expected = {
        "track_name": "Tampa Bay Downs",
        "race_type_normalized": "maiden claiming",
        "purse_amount": 16000,
    }

    passed = True
    for key, expected_value in expected.items():
        actual_value = result.get(key)
        if actual_value == expected_value:
            print(f"  ✅ {key}: {actual_value}")
        else:
            print(f"  ❌ {key}: Expected '{expected_value}', got '{actual_value}'")
            passed = False

    # Check race type detection
    if detected_type == expected["race_type_normalized"]:
        print(f"  ✅ detect_race_type: {detected_type}")
    else:
        print(
            f"  ❌ detect_race_type: Expected '{expected['race_type_normalized']}', got '{detected_type}'"
        )
        passed = False

    print("\n" + "=" * 80)
    if passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
