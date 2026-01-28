# Equibase Manual Navigation Guide
## Step-by-Step Instructions for Accessing Historical Race Data

⚠️ **CRITICAL REALITY CHECK** ⚠️

Based on extensive research, **Equibase does NOT offer free bulk historical data downloads** (2010-2025 range). Here's what's actually available:

---

## What's Actually Free vs Commercial

### ✅ **FREE (No Login Required)**
- Individual race charts (one race at a time, manual lookup)
- Today's entries/results (current racing day only)
- Basic horse/jockey/trainer profiles

### ❌ **COMMERCIAL ONLY (Requires Paid Subscription)**
- Bulk historical database (2010-2025)
- Past performances (PPs) export
- Multi-race CSV/database downloads
- API access
- Historical chart archives

**Cost**: $500-5,000/month depending on data scope (see [EQUIBASE_COMMERCIAL_RESEARCH.md](EQUIBASE_COMMERCIAL_RESEARCH.md))

---

## Option 1: Free Individual Race Charts (Manual, Time-Intensive)

### **If You Want to Try Manual Downloading:**

#### Step 1: Access Equibase Website
```
URL: https://www.equibase.com
Browser Requirements:
  - Enable JavaScript (required)
  - Enable Cookies (required)
  - Disable ad blockers (may interfere)
  - Use Chrome/Firefox (best compatibility)
```

**If You See "Pardon Our Interruption" Page:**
```
Solutions:
1. Wait 10-15 seconds (page auto-loads)
2. Clear browser cache: Ctrl+Shift+Delete
3. Open in Incognito/Private window
4. Try different browser
5. Disable VPN if using one
```

---

#### Step 2: Navigate to Race Charts

**Method A: Via Results Menu**
```
1. Go to: https://www.equibase.com
2. Top menu: Click "Results" dropdown
3. Select: "Chart Archive" or "Results Charts"
4. Should land on URL like:
   https://www.equibase.com/static/chart/
   OR
   https://www.equibase.com/results/results.cfm
```

**Method B: Direct URL Attempts** (may require adjustments)
```
Try these URLs:
- https://www.equibase.com/results/results.cfm
- https://www.equibase.com/static/chart/
- https://www.equibase.com/premium/eqbChartCalendar.cfm (may be premium-only)
- https://www.equibase.com/profiles/Results.cfm

Note: Exact URLs change periodically. Use site navigation if direct links fail.
```

---

#### Step 3: Filter to US Tracks Only

**Track Selection Interface:**
```
1. Look for "Track" or "Select Track" dropdown
2. US Tracks to Focus On:
   ✅ Gulfstream Park (GP) - Florida
   ✅ Santa Anita (SA) - California
   ✅ Churchill Downs (CD) - Kentucky
   ✅ Del Mar (DMR) - California
   ✅ Belmont Park (BEL) - New York
   ✅ Saratoga (SAR) - New York
   ✅ Keeneland (KEE) - Kentucky
   
3. AVOID Non-US Tracks:
   ❌ Woodbine (WO) - Canada
   ❌ Any track with country code outside USA
```

**Verification:**
```
- Check track abbreviation matches US list above
- Look for "USA" or state name in track details
- Verify purse is in USD (not CAD)
```

---

#### Step 4: Date Range Selection

**Limitations:**
```
⚠️ FREE VERSION: Typically only allows recent races (30-90 days)
⚠️ HISTORICAL (2010-2025): Requires premium subscription

Expected Interface:
1. Calendar picker: Select single date
2. Or: "Last 30 Days" / "Last Week" options
3. Historical years (2010-2020): Grayed out or requires login
```

**What You Can Realistically Access:**
```
✅ Last 30-90 days: Free
⚠️ Last 6 months: May require free account creation
❌ 2010-2025 bulk: Commercial subscription only
```

---

#### Step 5: Download Individual Race Charts

**For Each Race:**
```
1. Click race number (e.g., "Race 5")
2. View full chart with:
   - Finishing positions (1st through last)
   - Horse names, jockeys, trainers
   - Running times, margins
   - Comments (trouble notes)
   
3. Download Options:
   📄 PDF: Look for "Print" or "PDF" button
   📊 No CSV: Equibase free version does NOT offer CSV export
   
4. If PDF Available:
   - Right-click → Save As
   - Save to: C:\Users\C Stephens\Desktop\Horse Racing Picks\equibase_charts\
   - Name format: TRACK_DATE_RACE#.pdf (e.g., GP_20250128_R5.pdf)
```

**Manual Data Extraction:**
```
From PDF, manually record:
- Date, Track, Race Number
- Horse Name, Post Position
- Finish Position (1st, 2nd, 3rd, 4th, etc.)
- Jockey, Trainer
- Final Time, Beyer Speed Figure (if shown)

Save to spreadsheet or text file.
```

---

#### Step 6: US-Only Data Validation

**Before Processing Each File, Verify:**

```python
# Validation Checklist:
✅ Track abbreviation in US list (GP, SA, CD, DMR, BEL, SAR, KEE)
✅ Track location shows US state (not "Ontario" or other country)
✅ Purse shown in USD (not CAD or other currency)
✅ Date format matches US (MM/DD/YYYY, not DD/MM/YYYY)

# Red Flags (Non-US):
❌ Track: Woodbine (WO) = Canada
❌ Track: Hastings (HST) = Canada
❌ Any track with "(CAN)" suffix
❌ Purse in CAD (Canadian Dollars)
```

**Automated Check (if you have CSV):**
```python
import pandas as pd

# US track codes
US_TRACKS = ['GP', 'SA', 'CD', 'DMR', 'BEL', 'SAR', 'KEE', 'PIM', 
             'AQU', 'HOO', 'PRX', 'DEL', 'MTH', 'LRL', 'CT', 'TAM']

df = pd.read_csv('equibase_download.csv')

# Filter US only
us_df = df[df['Track'].isin(US_TRACKS)]

print(f"Total races: {len(df)}")
print(f"US races: {len(us_df)}")
print(f"Non-US filtered: {len(df) - len(us_df)}")
```

---

## Option 2: Premium Subscription (If You Want Bulk Access)

### **Equibase Premium/TrackMaster Access**

**If you see "Premium" or "TrackMaster" options:**

#### Step 1: Check Subscription Status
```
1. Login: https://www.equibase.com
2. Top-right: Click account icon
3. Check: "Subscription Status" or "My Account"
4. Look for: "Premium Member" or "TrackMaster Access"

If NOT subscribed:
- Free version: Very limited data
- Need to purchase: $500-5,000/month (commercial pricing)
```

#### Step 2: Navigate to Premium Archives (If Subscribed)
```
URLs to try:
- https://www.equibase.com/premium/eqbChartCalendar.cfm
- https://www.equibase.com/premium/eqbResultsCharts.cfm
- https://www.equibase.com/premium/trackmaster/

Expected Features:
- Date range selector (multi-year)
- Bulk export (CSV, Excel)
- Advanced filtering (class, surface, distance)
```

#### Step 3: Bulk Download (Premium Only)
```
1. Select Track(s): Multi-select US tracks
2. Date Range: 2010-2025 (if premium allows)
3. Export Format:
   ✅ CSV (best for ML)
   ✅ Excel (.xlsx)
   ⚠️ PDF (requires OCR parsing)
   
4. Click "Export" or "Download"
5. Wait for file generation (may take 5-30 minutes for large ranges)
6. Download to: C:\Users\C Stephens\Desktop\Horse Racing Picks\equibase_bulk\
```

---

## Option 3: Contact Equibase for Commercial Data (RECOMMENDED)

**If you need 2010-2025 bulk historical data, you MUST contact Equibase directly:**

### **Commercial Data Request**

**Email Template:**
```
To: info@equibase.com
Subject: Historical Data Request - Personal Use

Hello Equibase Team,

I'm interested in purchasing historical race data for machine learning 
analysis (personal, non-commercial use).

Data Requirements:
- Tracks: US only (Gulfstream, Santa Anita, Churchill Downs, etc.)
- Date Range: 2010-2025 (15 years)
- Data Fields Needed:
  * Race date, track, race number
  * Horse names, post positions
  * Finishing positions (1st through last)
  * Speed figures (Beyer if available)
  * Past performances
  * Jockey/trainer statistics

Export Format: CSV or database dump (PostgreSQL/MySQL compatible)

Questions:
1. What is the cost for this historical dataset?
2. One-time purchase or subscription?
3. Data delivery method (download link, FTP, API)?
4. Included data fields and schema?
5. Update frequency (if subscription)?

Please provide pricing and next steps.

Thank you,
[Your Name]
[Your Email]
[Your Phone]
```

**Or Call:**
```
Phone: 859-224-2800
Hours: 8am-5pm ET, Monday-Friday
Ask for: "Data Licensing" or "Commercial Data Sales"
```

**Expected Response:**
```
- Pricing quote: $500-5,000/month (or one-time $5K-25K)
- Sample data: May provide small sample to test
- Contract: Data usage agreement required
- Timeline: 1-4 weeks from agreement to data delivery
```

---

## Realistic Assessment: What You Can Actually Get

### **Free Manual Approach (Time-Intensive)**
```
Method: Individual race chart downloads
Effort: 5-10 minutes per race
Scale: ~10-50 races max (not practical for 1,000+)
Format: PDF (requires manual data entry or OCR)
Cost: $0 (just your time)

Verdict: ❌ Not viable for ML training (insufficient data)
```

### **Commercial Bulk Purchase (Expensive)**
```
Method: Equibase commercial subscription
Effort: 1-2 hours setup, instant data delivery
Scale: 10,000-100,000+ races
Format: CSV/database (ML-ready)
Cost: $500-5,000/month or $5K-25K one-time

Verdict: ✅ Only way to get 2010-2025 bulk data
         ⚠️ But expensive for personal use
```

### **Your Existing Solution (BEST)**
```
Method: Section F daily captures + BRISNET PPs
Effort: 1 minute/day (30 sec capture + 30 sec results)
Scale: 1,000+ races in 100 days
Format: SQLite database (ML-ready)
Cost: $0 additional (uses existing BRISNET purchases)

Verdict: 🏆 Superior for your use case
         ✅ Zero cost, sustainable, owns data
```

---

## Step-by-Step: If You Download ANY File

### **After Obtaining Data (CSV/PDF/Excel):**

#### Step 1: Save File
```
Location: C:\Users\C Stephens\Desktop\Horse Racing Picks\equibase_downloads\
Filename format: equibase_TRACK_STARTDATE_ENDDATE.csv
Example: equibase_GP_20230101_20231231.csv
```

#### Step 2: Initial Validation
```powershell
# Check file in PowerShell
cd "C:\Users\C Stephens\Desktop\Horse Racing Picks"

# View first 10 lines
Get-Content equibase_downloads\equibase_GP_20230101_20231231.csv | Select-Object -First 10

# Check file size
Get-ChildItem equibase_downloads\*.csv | Select-Object Name, Length
```

#### Step 3: Python Validation Script
```python
# Save as: validate_equibase_download.py
import pandas as pd

# Load file
file_path = 'equibase_downloads/equibase_GP_20230101_20231231.csv'
df = pd.read_csv(file_path)

print("="*70)
print("EQUIBASE DATA VALIDATION")
print("="*70)
print(f"\nFile: {file_path}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check for US tracks
if 'Track' in df.columns:
    us_tracks = ['GP', 'SA', 'CD', 'DMR', 'BEL', 'SAR', 'KEE']
    non_us = df[~df['Track'].isin(us_tracks)]
    print(f"\n⚠️  Non-US tracks found: {len(non_us)}")
    if len(non_us) > 0:
        print(f"   Tracks: {non_us['Track'].unique()}")

# Check for required columns
required = ['Date', 'Track', 'Horse', 'FinishPosition']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"\n❌ Missing columns: {missing}")
    print("   Available columns:", list(df.columns))
else:
    print(f"\n✅ All required columns present")

# Check finishing positions
if 'FinishPosition' in df.columns:
    print(f"\nFinishing Positions:")
    print(f"  1st place: {len(df[df['FinishPosition'] == 1])}")
    print(f"  2nd place: {len(df[df['FinishPosition'] == 2])}")
    print(f"  3rd place: {len(df[df['FinishPosition'] == 3])}")
    print(f"  4th place: {len(df[df['FinishPosition'] == 4])}")

print("\n" + "="*70)
print("Validation complete. Review output above.")
print("="*70)
```

#### Step 4: Upload for Processing
```
Once validated, notify in chat:
"I have downloaded [X] races from Equibase. 
File: equibase_downloads/equibase_TRACK_DATE.csv
Ready for integration into historical_data_builder.py"

Then I can create a parser specific to Equibase format.
```

---

## PDF Chart Parsing (If You Downloaded PDFs)

**If you have PDF race charts:**

```python
# Install OCR tools (if needed)
pip install pdfplumber pytesseract

# Save as: parse_equibase_pdf.py
import pdfplumber
import re

def parse_equibase_chart_pdf(pdf_path):
    """Extract race data from Equibase PDF chart."""
    
    with pdfplumber.open(pdf_path) as pdf:
        # Get first page (charts are usually 1 page)
        page = pdf.pages[0]
        text = page.extract_text()
        
        # Extract race info
        race_info = {}
        
        # Track and date (regex patterns - adjust based on actual format)
        track_match = re.search(r'([A-Z]{2,3})\s+Race\s+(\d+)', text)
        if track_match:
            race_info['track'] = track_match.group(1)
            race_info['race_num'] = track_match.group(2)
        
        # Extract finishing positions (look for table)
        horses = []
        lines = text.split('\n')
        
        for line in lines:
            # Look for pattern: Position Horse Jockey ...
            match = re.search(r'^(\d{1,2})\s+([A-Za-z\s\']+)\s+([A-Za-z\s]+)', line)
            if match:
                horses.append({
                    'finish_position': int(match.group(1)),
                    'horse_name': match.group(2).strip(),
                    'jockey': match.group(3).strip()
                })
        
        race_info['horses'] = horses
        return race_info

# Usage
result = parse_equibase_chart_pdf('equibase_downloads/GP_20250128_R5.pdf')
print(result)
```

---

## Summary: Your Best Path Forward

### **Reality Check**
```
❌ Equibase does NOT offer free bulk historical downloads (2010-2025)
❌ Manual race-by-race downloads are too time-intensive for ML
❌ Commercial subscriptions cost $500-5,000/month

✅ You already have a better solution built:
   - Section F in app.py (daily captures)
   - BRISNET integration (elite_parser.py)
   - SQLite database (historical_races.db)
   - Path to 90% in 100 days at $0 cost
```

### **Recommended Action**

**Today:**
```
1. ✅ Accept that free Equibase bulk data doesn't exist
2. ✅ Use Section F to start daily captures (you have this built!)
3. ✅ Download Kaggle Triple Crown dataset (45 free races)
4. ✅ Email BRISNET for bulk historical pricing (cheaper than Equibase)
```

**This Week:**
```
1. Capture 5-10 races/day via Section F
2. Wait for BRISNET bulk pricing response
3. Decide: Purchase BRISNET bulk ($500-1,500) or continue free path
```

**Month 1-4:**
```
1. Accumulate 1,000 races organically
2. Reach 90%+ accuracy
3. Own your data forever
4. Zero vendor lock-in
```

---

## If You STILL Want to Try Manual Equibase Access

### **Step-by-Step Checklist**

1. ☐ Visit https://www.equibase.com
2. ☐ Enable JavaScript & cookies in browser
3. ☐ Navigate to "Results" → "Chart Archive"
4. ☐ Select US track (GP, SA, CD, etc.)
5. ☐ Select recent date (last 30 days for free access)
6. ☐ Click individual race number
7. ☐ Download PDF or copy data manually
8. ☐ Verify US-only (check track code)
9. ☐ Repeat for each race (10-50 races max practical)
10. ☐ Upload files for processing

### **Expected Outcome**
```
Best Case: 10-50 races downloaded (1-3 hours effort)
Reality: Insufficient for ML training (need 1,000+ races)
Alternative: Use your Section F system (gets 1,000 races in 100 days)
```

---

## Contact Information

**If You Need Commercial Bulk Data:**

**Equibase:**
- Phone: 859-224-2800
- Email: info@equibase.com
- Best for: Enterprise-scale data (10,000+ races)

**BRISNET (Better for You):**
- Email: info@brisnet.com
- Best for: Existing customer bulk discount
- Use template in: [EQUIBASE_COMMERCIAL_RESEARCH.md](EQUIBASE_COMMERCIAL_RESEARCH.md)

---

## Bottom Line

**You're asking for something that doesn't exist for free.** Equibase's 2010-2025 bulk historical database requires commercial licensing ($500-5,000/month).

**You already built a superior solution:** Section F in app.py that captures real training data from your daily BRISNET workflow at zero additional cost.

**Start using what you built instead of chasing phantom free data.**

Navigate to Section F → Auto-Capture → Start capturing today 🚀
