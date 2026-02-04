"""
Check coverage of race type abbreviations in CLASS_MAP
"""
from race_class_parser import CLASS_MAP, LEVEL_MAP

# Common race type abbreviations used in North American horse racing
COMMON_ABBREVIATIONS = [
    # Basic graded
    'G1', 'G2', 'G3', 'GR1', 'GR2', 'GR3', 'GRADE1', 'GRADE2', 'GRADE3',
    
    # Stakes
    'STK', 'STAKES', 'S', 'N',
    
    # Handicap
    'HCP', 'HANDICAP', 'H',
    
    # Allowance
    'ALW', 'ALLOWANCE', 'A',
    
    # Claiming
    'CLM', 'CLAIMING', 'C', 'CL', 'CLG',
    
    # Maiden
    'MSW', 'MAIDEN', 'MDN', 'MD',
    'MCL', 'MDNCLM', 'MDC',
    
    # Conditional allowance (most critical)
    'AOC', 'AO',  # Allowance Optional Claiming
    'N1X', 'N2X', 'N3X',  # Non-winners of X
    'NW1', 'NW2', 'NW3',  # Alternative format
    'N1L', 'N2L', 'N3L',  # Non-winners lifetime
    'OC', 'OCL',  # Optional Claiming
    
    # Maiden variants
    'MOC',  # Maiden Optional Claiming
    'MSC',  # Maiden Starter Claiming
    
    # Starter
    'STA', 'STR', 'STARTER',
    'SOC',  # Starter Optional Claiming
    'CST', 'CLMSTK',  # Claiming Stakes
    
    # Special
    'L', 'LR', 'LISTED',
    'WCL', 'WAIVER',
    'TRL', 'TRIAL',
    'STB', 'STATEBRED',
    'OPT', 'OPTIONAL',
    
    # Specialty races (less common)
    'FUT', 'FUTURITY',  # Futurity
    'DER', 'DERBY',  # Derby
    'INVIT', 'INVITATIONAL',  # Invitational
]

def check_coverage():
    print("="*70)
    print("RACE TYPE ABBREVIATION COVERAGE CHECK")
    print("="*70)
    
    # Check what's missing
    missing = []
    for abbrev in COMMON_ABBREVIATIONS:
        if abbrev not in CLASS_MAP:
            missing.append(abbrev)
    
    # Check what's covered
    covered = []
    for abbrev in COMMON_ABBREVIATIONS:
        if abbrev in CLASS_MAP:
            covered.append(abbrev)
    
    print(f"\n✓ COVERED: {len(covered)}/{len(COMMON_ABBREVIATIONS)} abbreviations")
    print(f"⚠️  MISSING: {len(missing)}/{len(COMMON_ABBREVIATIONS)} abbreviations")
    
    if missing:
        print("\n" + "="*70)
        print("MISSING ABBREVIATIONS (Recommended to Add):")
        print("="*70)
        
        # Group by category
        conditional = [m for m in missing if 'NW' in m or 'N' in m and ('L' in m or 'X' in m)]
        optional = [m for m in missing if 'OC' in m or 'OPT' in m]
        specialty = [m for m in missing if m in ['FUT', 'FUTURITY', 'DER', 'DERBY', 'INVIT', 'INVITATIONAL']]
        other = [m for m in missing if m not in conditional + optional + specialty]
        
        if conditional:
            print("\n🎯 Conditional Allowance (HIGH PRIORITY):")
            for m in conditional:
                print(f"   • {m}")
                
        if optional:
            print("\n🎯 Optional Claiming Variants:")
            for m in optional:
                print(f"   • {m}")
                
        if other:
            print("\n📋 Other Missing:")
            for m in other:
                print(f"   • {m}")
                
        if specialty:
            print("\n🏆 Specialty Races (LOW PRIORITY):")
            for m in specialty:
                print(f"   • {m}")
    else:
        print("\n✅ ALL COMMON ABBREVIATIONS ARE COVERED!")
    
    print("\n" + "="*70)
    print("CURRENT SYSTEM STATISTICS")
    print("="*70)
    print(f"📊 Total abbreviations in CLASS_MAP: {len(CLASS_MAP)}")
    print(f"📊 Total hierarchy levels in LEVEL_MAP: {len(LEVEL_MAP)}")
    
    # Show hierarchy distribution
    print("\n📈 Hierarchy Level Distribution:")
    level_counts = {}
    for full_name, level in LEVEL_MAP.items():
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level in sorted(level_counts.keys(), reverse=True):
        print(f"   Level {level:2d}: {level_counts[level]:2d} class types")
    
    print("\n" + "="*70)
    
    return missing

if __name__ == "__main__":
    missing_abbrevs = check_coverage()
    
    if missing_abbrevs:
        print("\n💡 RECOMMENDATION:")
        print("   Add missing abbreviations to CLASS_MAP in race_class_parser.py")
        print("   This will ensure accurate weight calculations for all race types")
    else:
        print("\n🎉 System is ready! All common abbreviations covered.")
