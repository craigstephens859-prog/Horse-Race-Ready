# Type Annotations - IDE Support Examples

## What You Get Now

### 1. Method Signature Hints
When you type `engine._calc_`, your IDE will show:

```python
_calc_class(horse: HorseData, today_purse: int, today_race_type: str) -> float
_calc_form(horse: HorseData) -> float
_calc_speed(horse: HorseData, horses_in_race: List[HorseData]) -> float
_calc_pace(horse: HorseData, horses_in_race: List[HorseData], distance_txt: str) -> float
```

**Before:** Just method names with no type info  
**After:** Full signatures with parameter types and return types

### 2. Variable Autocomplete
When working with variables:

```python
# Your IDE now knows the exact type
rating: float = engine._calc_class(horse, 16500, "msw")
# Type: float (IDE shows this)

horses: List[HorseData] = parser.parse_full_pp(pp_text)
# Type: List[HorseData] (IDE autocompletes HorseData attributes)

result: pd.DataFrame = engine.predict_race(...)
# Type: DataFrame (IDE knows DataFrame methods available)
```

### 3. Error Detection
IDE will highlight errors before you run:

```python
# ❌ IDE shows error: Expected int, got str
rating = engine._calc_class(horse, "16500", "msw")

# ✅ Correct
rating = engine._calc_class(horse, 16500, "msw")

# ❌ IDE shows error: Expected List[HorseData], got HorseData
speed = engine._calc_speed(horse, horse)

# ✅ Correct
speed = engine._calc_speed(horse, [horse, horse2, horse3])
```

### 4. Return Type Awareness
IDE knows what you get back:

```python
# Unpacking with type safety
figs, avg, peak, last = parser._parse_speed_figures(block)
# figs: List[int]
# avg: float
# peak: int
# last: int

# Dictionary access with hints
pedigree: Dict[str, any] = parser._parse_pedigree(block)
sire_spi = pedigree.get('sire_spi')  # IDE knows this returns Optional[float]

# Optional handling
days: Optional[int] = horse.days_since_last
if days is not None:  # IDE understands type narrowing
    layoff_penalty = days / 30.0  # OK - days is int here
```

### 5. HorseData Attribute Access
Complete autocomplete for all 50+ fields:

```python
horse: HorseData = horses["Way of Appeal"]

# IDE shows all available attributes with types:
horse.post              # str
horse.quirin_points     # float
horse.jockey_win_pct    # float
horse.speed_figures     # List[int]
horse.days_since_last   # Optional[int]
horse.equipment_change  # Optional[str]
horse.surface_stats     # Dict
horse.first_lasix       # bool
```

### 6. Complex Return Types
Union types properly handled:

```python
# Debug mode returns tuple
angles_df, debug_df = compute_eight_angles(df, debug=True)
# angles_df: pd.DataFrame
# debug_df: pd.DataFrame

# Normal mode returns single DataFrame
angles_df = compute_eight_angles(df, debug=False)
# angles_df: pd.DataFrame
```

## Real-World Example

```python
from unified_rating_engine import UnifiedRatingEngine
from elite_parser import EliteBRISNETParser, HorseData
from typing import List, Dict

# Create instances (types inferred)
parser: EliteBRISNETParser = EliteBRISNETParser()
engine: UnifiedRatingEngine = UnifiedRatingEngine(softmax_tau=3.0)

# Parse PP text (IDE knows return type)
pp_text: str = load_brisnet_pp()
horses: Dict[str, HorseData] = parser.parse_full_pp(pp_text)

# Validate with type safety
validation: Dict[str, any] = parser.validate_parsed_data(horses)
confidence: float = validation['overall_confidence']

if confidence >= 0.7:
    # Get predictions (IDE knows DataFrame returned)
    results: pd.DataFrame = engine.predict_race(
        pp_text=pp_text,
        today_purse=16500,  # int required
        today_race_type="msw",  # str required
        track_name="Mountaineer",
        surface_type="Dirt",
        distance_txt="5.5 Furlongs"
    )
    
    # Access DataFrame columns (IDE autocompletes)
    winner: str = results.iloc[0]['Horse']
    win_prob: float = results.iloc[0]['Probability']
    
    print(f"✅ Top pick: {winner} ({win_prob:.1%})")
```

**IDE Benefits:**
- ✅ Autocomplete at every step
- ✅ Type errors highlighted before running
- ✅ Return types clearly shown
- ✅ Parameter hints with correct types
- ✅ No need to check documentation constantly

## Static Type Checking with mypy

Run comprehensive type checking:

```bash
# Install mypy
pip install mypy

# Check all core files
mypy unified_rating_engine.py elite_parser.py horse_angles8.py

# Expected output:
# Success: no issues found in 3 source files
```

### Enable Strict Mode (Optional)

Create `mypy.ini`:
```ini
[mypy]
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
```

## VS Code Integration

### Pylance Settings (Recommended)

Add to `.vscode/settings.json`:
```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.completeFunctionParens": true,
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.inlayHints.functionReturnTypes": true,
    "python.analysis.inlayHints.variableTypes": true
}
```

**What You Get:**
- Inline type hints next to variables
- Function return type hints
- Parameter type hints on hover
- Instant error highlighting
- Smart import completions

### Example VS Code Experience

When you hover over a function:
```
_calc_class(horse: HorseData, today_purse: int, today_race_type: str) -> float

Class rating: purse comparison + race type hierarchy

Parameters:
  horse: HorseData - Horse data object with all fields
  today_purse: int - Today's purse amount
  today_race_type: str - Race classification (msw, alw, stk, etc.)

Returns:
  float - Class rating [-3.0 to +6.0]
```

## Common Patterns

### Working with Optional Types

```python
# Optional[int] handling
days: Optional[int] = horse.days_since_last

# Bad (IDE warns)
penalty = days / 30.0  # Error: Optional[int] not compatible with division

# Good (IDE approves)
if days is not None:
    penalty = days / 30.0  # OK: type narrowed to int
else:
    penalty = 0.0
```

### Working with List Types

```python
# List[int] operations
figs: List[int] = horse.speed_figures

# IDE knows all list methods
if figs:
    top_fig: int = max(figs)  # IDE infers int
    avg_fig: float = sum(figs) / len(figs)
```

### Working with Dict Types

```python
# Dict[str, any] access
stats: Dict[str, any] = horse.surface_stats.get('Fst', {})

# IDE helps with get() default
win_pct: float = stats.get('win_pct', 0.0)
starts: int = stats.get('starts', 0)
```

## Testing Type Annotations

Verify signatures are accessible:

```python
import inspect
from unified_rating_engine import UnifiedRatingEngine

engine = UnifiedRatingEngine()
sig = inspect.signature(engine._calc_class)
print(sig)
# Output: (horse: elite_parser.HorseData, today_purse: int, 
#          today_race_type: str) -> float
```

## Performance Impact

**None.** Type annotations are:
- ✅ Compile-time only (Python 3.5+)
- ✅ No runtime overhead
- ✅ No performance impact
- ✅ Completely optional for execution

Python ignores type hints at runtime, so all existing code runs identically.

## Summary

Type annotations transform your coding experience from:
- ❌ Guessing return types
- ❌ Looking up documentation constantly
- ❌ Runtime type errors
- ❌ No autocomplete support

To:
- ✅ IDE shows everything
- ✅ Self-documenting code
- ✅ Compile-time error detection
- ✅ Full autocomplete everywhere

**Your code is now professional-grade with enterprise-level type safety!**
