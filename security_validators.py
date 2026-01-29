"""
Security Input Validation Module
=================================

Provides sanitization and validation for all user inputs to prevent:
- SQL injection attacks
- Code injection (eval/exec)
- Path traversal attacks
- Denial of service (oversized inputs)
- Unicode attacks
- Format string attacks

Author: Security Review - January 29, 2026
"""

import re
import html
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =====================================================
# SIZE LIMITS (Prevent DoS)
# =====================================================
MAX_PP_TEXT_SIZE = 100_000  # 100KB
MAX_TRACK_NAME_LENGTH = 50
MAX_HORSE_NAME_LENGTH = 100
MAX_RACE_ID_LENGTH = 50

# =====================================================
# ALLOWED PATTERNS
# =====================================================
TRACK_CODE_PATTERN = re.compile(r'^[A-Z]{2,5}$')
RACE_ID_PATTERN = re.compile(r'^[A-Z0-9_\-]{3,50}$')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DISTANCE_SAFE_PATTERN = re.compile(r'^[\d\s./Mmilefurlongs]+$', re.IGNORECASE)

# =====================================================
# SQL INJECTION PATTERNS
# =====================================================
SQL_INJECTION_PATTERNS = [
    re.compile(r';.*?DROP', re.IGNORECASE),
    re.compile(r';.*?DELETE', re.IGNORECASE),
    re.compile(r';.*?TRUNCATE', re.IGNORECASE),
    re.compile(r'UNION.*?SELECT', re.IGNORECASE),
    re.compile(r'--.*$', re.MULTILINE),
    re.compile(r'/\*.*?\*/', re.DOTALL),
    re.compile(r'exec\s*\(', re.IGNORECASE),
    re.compile(r'execute\s*\(', re.IGNORECASE),
]

# =====================================================
# CODE INJECTION PATTERNS
# =====================================================
CODE_INJECTION_PATTERNS = [
    re.compile(r'__import__', re.IGNORECASE),
    re.compile(r'eval\s*\(', re.IGNORECASE),
    re.compile(r'exec\s*\(', re.IGNORECASE),
    re.compile(r'compile\s*\(', re.IGNORECASE),
    re.compile(r'open\s*\(', re.IGNORECASE),
    re.compile(r'os\.system', re.IGNORECASE),
    re.compile(r'subprocess', re.IGNORECASE),
]

# =====================================================
# WHITELISTS
# =====================================================
ALLOWED_SQL_TABLES = {
    'races_analyzed',
    'gold_high_iq',
    'races',
    'runners',
    'results',
    'pp_lines',
    'metadata'
}

ALLOWED_FILE_EXTENSIONS = {'.txt', '.csv', '.json', '.md'}


# =====================================================
# PP TEXT VALIDATION
# =====================================================

def sanitize_pp_text(text: str) -> str:
    """
    Sanitize BRISNET PP text input
    
    Security checks:
    1. Size limit (prevent DoS)
    2. Remove control characters
    3. Detect SQL injection patterns
    4. Detect code injection patterns
    5. Normalize whitespace
    
    Args:
        text: Raw PP text from user
        
    Returns:
        Sanitized PP text
        
    Raises:
        ValueError: If text contains malicious patterns or exceeds size limit
    """
    # 1. Size validation
    if len(text) > MAX_PP_TEXT_SIZE:
        raise ValueError(f"PP text too large (max {MAX_PP_TEXT_SIZE} chars, got {len(text)})")
    
    if not text or not text.strip():
        raise ValueError("PP text cannot be empty")
    
    # 2. Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 3. Detect SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(f"Suspicious SQL pattern detected: {pattern.pattern}")
            raise ValueError("Invalid characters detected in PP text")
    
    # 4. Detect code injection patterns
    for pattern in CODE_INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(f"Suspicious code pattern detected: {pattern.pattern}")
            raise ValueError("Invalid characters detected in PP text")
    
    # 5. Normalize excessive whitespace (but keep newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def validate_distance_string(distance: str) -> str:
    """
    Validate distance string before parsing
    
    Prevents code injection via eval() in distance parsing
    
    Args:
        distance: Distance string like "6F", "1 1/16M", "8.5 furlongs"
        
    Returns:
        Validated distance string
        
    Raises:
        ValueError: If distance contains unsafe characters
    """
    if not distance or len(distance) > 50:
        raise ValueError("Invalid distance string")
    
    # Only allow safe characters for distance
    if not DISTANCE_SAFE_PATTERN.match(distance):
        raise ValueError(f"Invalid distance format: {distance}")
    
    return distance.strip()


# =====================================================
# RACE METADATA VALIDATION
# =====================================================

def sanitize_race_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and validate race metadata
    
    Args:
        metadata: Race metadata dictionary
        
    Returns:
        Sanitized metadata
        
    Raises:
        ValueError: If critical fields are invalid
    """
    sanitized = {}
    
    # Validate race_id (required)
    race_id = str(metadata.get('race_id', '')).strip()
    if not RACE_ID_PATTERN.match(race_id):
        raise ValueError(f"Invalid race_id format: {race_id}")
    sanitized['race_id'] = race_id
    
    # Validate track code (3-5 letters)
    track = str(metadata.get('track', 'UNK')).strip().upper()
    if not TRACK_CODE_PATTERN.match(track):
        logger.warning(f"Invalid track code '{track}', using 'UNK'")
        track = 'UNK'
    sanitized['track'] = track
    
    # Validate date (YYYY-MM-DD)
    date = str(metadata.get('date', '')).strip()
    if not DATE_PATTERN.match(date):
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
        logger.warning(f"Invalid date, using today: {date}")
    sanitized['date'] = date
    
    # Validate numeric fields (must be non-negative)
    numeric_fields = ['purse', 'race_num', 'field_size', 'distance_furlongs']
    for field in numeric_fields:
        value = metadata.get(field, 0)
        try:
            value = float(value)
            if value < 0:
                value = 0
        except (ValueError, TypeError):
            value = 0
        sanitized[field] = value
    
    # Sanitize text fields (remove SQL injection patterns)
    text_fields = ['race_name', 'race_type', 'surface', 'conditions']
    for field in text_fields:
        value = str(metadata.get(field, '')).strip()
        # Remove dangerous characters
        value = re.sub(r'[;\'\"\\]', '', value)
        value = value[:200]  # Limit length
        sanitized[field] = value
    
    return sanitized


def validate_track_name(track: str) -> str:
    """
    Validate track name from user input
    
    Args:
        track: Track name
        
    Returns:
        Validated track name
        
    Raises:
        ValueError: If track name is invalid
    """
    track = str(track).strip()
    
    if not track or len(track) > MAX_TRACK_NAME_LENGTH:
        raise ValueError(f"Track name must be 1-{MAX_TRACK_NAME_LENGTH} characters")
    
    # Allow alphanumeric, spaces, hyphens, apostrophes only
    if not re.match(r'^[A-Za-z0-9\s\-\']{1,50}$', track):
        raise ValueError("Track name contains invalid characters")
    
    # Check for SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(track):
            raise ValueError("Invalid track name")
    
    return track


def validate_horse_name(name: str) -> str:
    """
    Validate horse name
    
    Args:
        name: Horse name
        
    Returns:
        Validated horse name
        
    Raises:
        ValueError: If name is invalid
    """
    name = str(name).strip()
    
    if not name or len(name) > MAX_HORSE_NAME_LENGTH:
        raise ValueError(f"Horse name must be 1-{MAX_HORSE_NAME_LENGTH} characters")
    
    # Allow alphanumeric, spaces, hyphens, apostrophes only
    if not re.match(r'^[A-Za-z0-9\s\-\'\.]{1,100}$', name):
        raise ValueError("Horse name contains invalid characters")
    
    return name


def validate_program_number(num: Any) -> int:
    """
    Validate program number
    
    Args:
        num: Program number (1-20)
        
    Returns:
        Validated program number
        
    Raises:
        ValueError: If number is out of range
    """
    try:
        num = int(num)
    except (ValueError, TypeError):
        raise ValueError("Program number must be an integer")
    
    if not 1 <= num <= 20:
        raise ValueError("Program number must be 1-20")
    
    return num


# =====================================================
# DATABASE QUERY VALIDATION
# =====================================================

def validate_table_name(table: str) -> str:
    """
    Validate SQL table name against whitelist
    
    Args:
        table: Table name
        
    Returns:
        Validated table name
        
    Raises:
        ValueError: If table is not in whitelist
    """
    table = str(table).strip().lower()
    
    if table not in ALLOWED_SQL_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    
    return table


def validate_sql_limit(limit: Any, max_limit: int = 100) -> int:
    """
    Validate SQL LIMIT clause value
    
    Args:
        limit: Requested limit
        max_limit: Maximum allowed limit
        
    Returns:
        Validated limit value
        
    Raises:
        ValueError: If limit is invalid
    """
    try:
        limit = int(limit)
    except (ValueError, TypeError):
        raise ValueError("Limit must be an integer")
    
    if limit < 1:
        raise ValueError("Limit must be positive")
    
    if limit > max_limit:
        logger.warning(f"Limit {limit} exceeds max {max_limit}, using {max_limit}")
        limit = max_limit
    
    return limit


def validate_sql_offset(offset: Any) -> int:
    """
    Validate SQL OFFSET clause value
    
    Args:
        offset: Requested offset
        
    Returns:
        Validated offset value
        
    Raises:
        ValueError: If offset is invalid
    """
    try:
        offset = int(offset)
    except (ValueError, TypeError):
        raise ValueError("Offset must be an integer")
    
    if offset < 0:
        raise ValueError("Offset cannot be negative")
    
    return offset


# =====================================================
# FILE PATH VALIDATION
# =====================================================

def validate_file_path(filepath: str, base_dir: str = ".", allowed_extensions: set = None) -> Path:
    """
    Validate file path for safe file operations
    
    Prevents:
    - Path traversal attacks (../)
    - Absolute path injection
    - Writing outside base directory
    
    Args:
        filepath: Requested file path
        base_dir: Base directory (default: current)
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is unsafe
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_FILE_EXTENSIONS
    
    # Resolve absolute paths
    base_path = Path(base_dir).resolve()
    target_path = (base_path / filepath).resolve()
    
    # Check for path traversal
    if not str(target_path).startswith(str(base_path)):
        raise ValueError(f"Path traversal detected: {filepath}")
    
    # Validate file extension
    if target_path.suffix not in allowed_extensions:
        raise ValueError(f"Invalid file extension: {target_path.suffix}")
    
    # Check filename length
    if len(target_path.name) > 255:
        raise ValueError("Filename too long")
    
    return target_path


# =====================================================
# HTML/XSS SANITIZATION (for Streamlit display)
# =====================================================

def sanitize_html_display(text: str) -> str:
    """
    Sanitize text for safe HTML display
    
    Prevents XSS attacks in Streamlit markdown/html
    
    Args:
        text: Text to display
        
    Returns:
        HTML-escaped text
    """
    return html.escape(str(text))


# =====================================================
# RATE LIMITING HELPERS
# =====================================================

class RateLimiter:
    """
    Simple rate limiter for API calls and expensive operations
    
    Usage:
        limiter = RateLimiter(max_calls=10, time_window=60)
        if limiter.allow_call():
            # Proceed with operation
        else:
            # Rate limit exceeded
    """
    
    def __init__(self, max_calls: int = 10, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def allow_call(self) -> bool:
        """
        Check if call is allowed under rate limit
        
        Returns:
            True if call allowed, False if rate limit exceeded
        """
        import time
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            return False
        
        self.calls.append(now)
        return True
    
    def reset(self):
        """Reset rate limiter"""
        self.calls = []


# =====================================================
# USAGE EXAMPLES
# =====================================================

if __name__ == "__main__":
    # Example 1: Sanitize PP text
    try:
        safe_text = sanitize_pp_text("RACE 1\nHORSE DATA...")
        print(f"✅ PP text validated: {len(safe_text)} chars")
    except ValueError as e:
        print(f"❌ PP text rejected: {e}")
    
    # Example 2: Validate track name
    try:
        track = validate_track_name("Gulfstream Park")
        print(f"✅ Track validated: {track}")
    except ValueError as e:
        print(f"❌ Track rejected: {e}")
    
    # Example 3: Validate SQL table name
    try:
        table = validate_table_name("races_analyzed")
        print(f"✅ Table validated: {table}")
    except ValueError as e:
        print(f"❌ Table rejected: {e}")
    
    # Example 4: Rate limiter
    limiter = RateLimiter(max_calls=3, time_window=5)
    for i in range(5):
        if limiter.allow_call():
            print(f"✅ Call {i+1} allowed")
        else:
            print(f"❌ Call {i+1} rate limited")
    
    # Example 5: Sanitize race metadata
    metadata = {
        'race_id': 'GP_20260129_R1',
        'track': 'GP',
        'date': '2026-01-29',
        'purse': 50000,
        'field_size': 12
    }
    safe_metadata = sanitize_race_metadata(metadata)
    print(f"✅ Metadata validated: {safe_metadata}")
