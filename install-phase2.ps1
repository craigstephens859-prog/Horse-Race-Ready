# Phase 2 ML System Installation Script
# Run this in PowerShell

Write-Host "🏇 Horse Racing Picks - Phase 2 ML System Installer" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
$requiredFile = ".streamlit\streamlit_app.py"

if (-not (Test-Path $requiredFile)) {
    Write-Host "❌ Error: Must run from 'Horse Racing Picks' directory" -ForegroundColor Red
    Write-Host "Current location: $currentDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run these commands:" -ForegroundColor Yellow
    Write-Host '  cd "C:\Users\C Stephens\Desktop\Horse Racing Picks"' -ForegroundColor White
    Write-Host "  .\install-phase2.ps1" -ForegroundColor White
    exit 1
}

Write-Host "✅ Correct directory confirmed" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Choose ML backend:" -ForegroundColor Cyan
Write-Host "  1. PyTorch (Recommended - GPU support)" -ForegroundColor White
Write-Host "  2. Scikit-learn (Lighter - CPU only)" -ForegroundColor White
Write-Host "  3. Install both (Maximum compatibility)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow

switch ($choice) {
    "1" {
        Write-Host "Installing PyTorch..." -ForegroundColor Cyan
        python -m pip install --upgrade pip
        python -m pip install torch torchvision pandas numpy streamlit
    }
    "2" {
        Write-Host "Installing Scikit-learn..." -ForegroundColor Cyan
        python -m pip install --upgrade pip
        python -m pip install scikit-learn pandas numpy streamlit
    }
    "3" {
        Write-Host "Installing both backends..." -ForegroundColor Cyan
        python -m pip install --upgrade pip
        python -m pip install torch torchvision scikit-learn pandas numpy streamlit
    }
    default {
        Write-Host "❌ Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Yellow

# Test ML engine import
$testScript = @"
try:
    from ml_engine import MLCalibrator, RaceDatabase
    print('✅ ML Engine loaded successfully')
except Exception as e:
    print(f'❌ ML Engine error: {e}')

try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
except:
    print('ℹ️  PyTorch not installed (will use sklearn)')

try:
    import sklearn
    print(f'✅ Scikit-learn version: {sklearn.__version__}')
except:
    print('ℹ️  Scikit-learn not installed (will use torch)')

# Test database creation
try:
    db = RaceDatabase()
    print(f'✅ Database initialized: {db.db_path}')
except Exception as e:
    print(f'❌ Database error: {e}')
"@

$testScript | python

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Installation complete! 🎉" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run the app: streamlit run .streamlit\streamlit_app.py" -ForegroundColor White
Write-Host "  2. Make predictions on races" -ForegroundColor White
Write-Host "  3. Enter results after races complete" -ForegroundColor White
Write-Host "  4. Train ML model after 50+ races" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: PHASE2_SETUP_GUIDE.md" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Cyan
