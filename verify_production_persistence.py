"""
PRODUCTION PERSISTENCE VERIFICATION
====================================
Verifies your Render deployment has proper data persistence configured.

Run this locally to check your production environment setup.
"""

import os
import sys

print("=" * 80)
print("PRODUCTION PERSISTENCE VERIFICATION")
print("=" * 80)

# Import the persistence layer
try:
    from db_persistence import (
        _get_github_config,
        get_persistence_status,
        has_persistent_disk,
        is_render,
    )

    PERSISTENCE_AVAILABLE = True
except ImportError:
    print("❌ ERROR: db_persistence.py not found!")
    sys.exit(1)

# Check local database status
print("\n📊 LOCAL DATABASE STATUS:")
print("-" * 80)

try:
    import sqlite3

    db_path = "gold_high_iq.db"

    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"✅ Database exists: {db_path}")
        print(f"   Size: {size_mb:.2f} MB")

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Get table counts
        c.execute("SELECT COUNT(*) FROM races_analyzed")
        races = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM race_results_summary")
        results = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM horses_analyzed")
        horses = c.fetchone()[0]

        conn.close()

        print(
            f"   Data: {races} races analyzed, {results} with results, {horses} horses"
        )
    else:
        print(f"⚠️  Database not found: {db_path}")
        print("   (This is normal if you haven't analyzed any races yet)")

except Exception as e:
    print(f"❌ Error checking database: {e}")

# Check persistence configuration
print("\n🔧 PERSISTENCE CONFIGURATION:")
print("-" * 80)

# Check GitHub backup configuration
token, repo = _get_github_config()

print("\n1️⃣  GITHUB BACKUP:")
if token and repo:
    print(f"   ✅ GITHUB_TOKEN: Set (length: {len(token)} chars)")
    print(f"   ✅ GITHUB_REPO: {repo}")
    print("\n   📝 What this means:")
    print("   • Database automatically backed up to GitHub after each race")
    print("   • Data restored from GitHub if Render redeploys")
    print("   • Backup branch: db-backup")
    print("   • Survives: ✅ Redeploys, ✅ Service deletion, ✅ Complete disaster")
else:
    print("   ⚠️  GITHUB_TOKEN: NOT SET")
    print("   ⚠️  GITHUB_REPO: NOT SET")
    print("\n   📝 What this means:")
    print("   • No automatic backups")
    print("   • Data could be lost on Render issues")
    print("   • Recommendation: Add these environment variables in Render Dashboard")

# Check for Render persistent disk capability
print("\n2️⃣  RENDER PERSISTENT DISK:")
print("   ℹ️  Cannot verify from local machine")
print("   ℹ️  Must check in Render Dashboard:")
print("      • Go to: https://dashboard.render.com")
print("      • Select your service")
print("      • Click 'Disks' tab")
print("      • Look for '/data' mount point")
print("\n   📝 What persistent disk provides:")
print("   • Fastest access (no GitHub API calls)")
print("   • Data survives redeploys")
print("   • Cost: $0.25/month per GB")
print("   • Does NOT survive: Service deletion, account issues")

# Current environment detection
print("\n🌍 CURRENT ENVIRONMENT:")
print("-" * 80)

if is_render():
    print("   📍 Running on: RENDER.COM (Production)")

    if has_persistent_disk():
        print("   ✅ Persistent disk: MOUNTED at /data")
        print("   💾 Persistence level: MAXIMUM (persistent disk + GitHub backup)")
    else:
        if token and repo:
            print("   ⚠️  Persistent disk: NOT MOUNTED")
            print("   ☁️  Persistence level: MEDIUM (GitHub backup only)")
        else:
            print("   ❌ Persistent disk: NOT MOUNTED")
            print("   ❌ GitHub backup: NOT CONFIGURED")
            print("   🚨 Persistence level: NONE (data lost on redeploy!)")
else:
    print("   💻 Running on: LOCAL MACHINE (Development)")
    print("   📝 To verify production, check logs after deploying to Render")

# Production readiness checklist
print("\n✅ PRODUCTION READINESS CHECKLIST:")
print("-" * 80)

ready_items = []
warning_items = []

if os.path.exists("gold_high_iq.db"):
    ready_items.append("✅ Local database exists (can be pushed to production)")
else:
    warning_items.append("⚠️  No local database yet (analyze races to create)")

if token and repo:
    ready_items.append("✅ GitHub backup configured (automatic saves)")
else:
    warning_items.append("⚠️  GitHub backup NOT configured")

ready_items.append("⚠️  Persistent disk: Check Render Dashboard → Disks")

for item in ready_items:
    print(f"   {item}")

for item in warning_items:
    print(f"   {item}")

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("-" * 80)

if not (token and repo):
    print("\n⚠️  PRIORITY: Configure GitHub Backup")
    print("   1. Create GitHub Personal Access Token:")
    print("      • Go to: https://github.com/settings/tokens")
    print("      • Click 'Generate new token (classic)'")
    print("      • Name: 'Render DB Backup'")
    print("      • Scopes: Check 'repo' (all sub-items)")
    print("      • Click 'Generate token'")
    print("      • Copy the token (you won't see it again!)")
    print()
    print("   2. Add to Render:")
    print("      • Go to: https://dashboard.render.com")
    print("      • Select your service")
    print("      • Click 'Environment' tab")
    print("      • Add environment variable:")
    print("        Key: GITHUB_TOKEN")
    print("        Value: ghp_xxxxxxxxxxxx (your token)")
    print("      • Add another environment variable:")
    print("        Key: GITHUB_REPO")
    print("        Value: craigstephens859-prog/Horse-Race-Ready")
    print("      • Save changes (triggers redeploy)")
    print()

print("\n📋 OPTIONAL: Add Persistent Disk (Faster + More Reliable)")
print("   1. Go to: https://dashboard.render.com")
print("   2. Select your service")
print("   3. Click 'Disks' tab")
print("   4. Click 'Add Disk'")
print("   5. Configure:")
print("      • Name: race-data")
print("      • Mount Path: /data")
print("      • Size: 1 GB ($0.25/month)")
print("   6. Click 'Create'")
print("   7. Service will redeploy automatically")
print()

print("\n🧪 TO VERIFY AFTER DEPLOYMENT:")
print("-" * 80)
print("   1. Deploy to Render")
print("   2. Visit your app: https://handicappinghorseraces.org/handicappingpicks")
print("   3. Scroll to 'E. Gold High-IQ System' section")
print("   4. Look for status message:")
print(
    "      • ✅ GOOD: '🔒 Data Persistence: All analyzed races are permanently saved'"
)
print("      • ⚠️  OK: '☁️ Data Persistence: Backed up to GitHub'")
print(
    "      • ❌ BAD: '⚠️ Data Persistence: Database is saved but on ephemeral storage'"
)
print()
print("=" * 80)
print("Verification complete!")
print("=" * 80)
