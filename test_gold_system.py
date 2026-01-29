"""
Test Script: Gold High-IQ System Verification

This script verifies that the Gold High-IQ System is properly installed
and all components are working correctly.

Run this BEFORE deploying to production.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_test(name, passed):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    return passed

def main():
    print_header("GOLD HIGH-IQ SYSTEM - VERIFICATION TESTS")
    
    all_passed = True
    
    # Test 1: Check file existence
    print_header("Test 1: File Existence")
    
    required_files = [
        "app.py",
        "gold_database_schema.sql",
        "gold_database_manager.py",
        "retrain_model.py",
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        all_passed &= print_test(f"File exists: {file}", exists)
        if exists:
            size_kb = Path(file).stat().st_size / 1024
            print(f"   └─ Size: {size_kb:.1f} KB")
    
    # Test 2: Check models directory
    print_header("Test 2: Models Directory")
    
    models_dir = Path("models")
    exists = models_dir.exists() and models_dir.is_dir()
    all_passed &= print_test("Models directory exists", exists)
    
    # Test 3: Import gold_database_manager
    print_header("Test 3: Import gold_database_manager")
    
    try:
        from gold_database_manager import GoldHighIQDatabase
        all_passed &= print_test("Import GoldHighIQDatabase", True)
        
        # Test initialization
        try:
            db = GoldHighIQDatabase(":memory:")  # Use in-memory for testing
            all_passed &= print_test("Initialize database (in-memory)", True)
            
            # Test schema creation
            import sqlite3
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            
            # Check if tables exist would be created
            with open("gold_database_schema.sql", "r", encoding="utf-8") as f:
                schema = f.read()
                cursor.executescript(schema)
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                "races_analyzed",
                "horses_analyzed",
                "gold_high_iq",
                "retraining_history",
                "race_results_summary"
            ]
            
            for table in expected_tables:
                exists = table in tables
                all_passed &= print_test(f"Table exists: {table}", exists)
            
            conn.close()
            
        except Exception as e:
            all_passed &= print_test("Initialize database", False)
            print(f"   └─ Error: {e}")
    
    except ImportError as e:
        all_passed &= print_test("Import GoldHighIQDatabase", False)
        print(f"   └─ Error: {e}")
    
    # Test 4: Import retrain_model
    print_header("Test 4: Import retrain_model")
    
    try:
        from retrain_model import RankingNN, plackett_luce_loss, retrain_model
        all_passed &= print_test("Import RankingNN", True)
        all_passed &= print_test("Import plackett_luce_loss", True)
        all_passed &= print_test("Import retrain_model", True)
        
        # Test PyTorch import
        try:
            import torch
            all_passed &= print_test("PyTorch available", True)
            print(f"   └─ PyTorch version: {torch.__version__}")
        except ImportError:
            all_passed &= print_test("PyTorch available", False)
            print("   └─ Warning: PyTorch not installed. Retraining will not work.")
    
    except ImportError as e:
        all_passed &= print_test("Import retrain_model", False)
        print(f"   └─ Error: {e}")
    
    # Test 5: Check app.py modifications
    print_header("Test 5: Check app.py Modifications")
    
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()
    
    checks = [
        ("Import time module", "import time" in app_content),
        ("Import GoldHighIQDatabase", "from gold_database_manager import GoldHighIQDatabase" in app_content),
        ("Initialize gold_db", "gold_db = GoldHighIQDatabase" in app_content),
        ("Auto-save after Analyze", "gold_db.save_analyzed_race" in app_content),
        ("Section E: Gold High-IQ", "Gold High-IQ System" in app_content or "Gold High-IQ" in app_content),
        ("Submit Actual Top 5", "Submit Actual Top 5" in app_content),
        ("Retrain Model tab", "Retrain Model" in app_content),
    ]
    
    for check_name, condition in checks:
        all_passed &= print_test(check_name, condition)
    
    # Test 6: Check documentation
    print_header("Test 6: Documentation Files")
    
    docs = [
        "GOLD_HIGH_IQ_IMPLEMENTATION_COMPLETE.md",
        "INTEGRATION_COMPLETE_GUIDE.md",
        "QUICKSTART_GOLD_HIGH_IQ.md",
        "IMPLEMENTATION_COMPLETE_SUMMARY.md",
        "ARCHITECTURE_DIAGRAM.md"
    ]
    
    for doc in docs:
        exists = Path(doc).exists()
        all_passed &= print_test(f"Doc exists: {doc}", exists)
    
    # Final summary
    print_header("FINAL RESULT")
    
    if all_passed:
        print("\n✅✅✅ ALL TESTS PASSED ✅✅✅")
        print("\nYour Gold High-IQ System is ready for deployment!")
        print("\nNext steps:")
        print("1. Run: python -m streamlit run app.py")
        print("2. Parse a race and click 'Analyze This Race'")
        print("3. Check for '💾 Auto-saved to gold database' message")
        print("4. Go to Section E and verify new UI")
        print("\nSee QUICKSTART_GOLD_HIGH_IQ.md for user guide.")
    else:
        print("\n❌ SOME TESTS FAILED ❌")
        print("\nPlease fix the failed tests before deploying.")
        print("Check the error messages above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
