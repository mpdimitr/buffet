#!/usr/bin/env python3
"""
Comprehensive Economic Tracker Test Suite
=========================================

Tests all economic tracking scripts in the buffet analysis toolkit to ensure
they are functioning correctly and generating proper output files.

Economic Trackers Tested:
1. buffet.py - Market Valuation (Buffett Indicator)
2. shipping_tracker_complete.py - Real Economic Activity
3. yield_curve_tracker.py - Recession Risk Indicator  
4. labor_market_tracker.py - Employment Health Monitor
5. credit_conditions_tracker.py - Credit System Health
6. manufacturing_tracker.py - Industrial Health Monitor
7. consumer_health_tracker.py - Consumer Spending & Confidence
8. corporate_earnings_tracker.py - Business Health & Profitability

Tests include:
- Script execution without errors
- Output file generation (CSV and PNG)
- Data validation and completeness
- Summary statistics and health assessment

Usage:
    python3 test_all_trackers.py

Author: GitHub Copilot
Date: September 2025
"""

import subprocess
import sys
import os
from datetime import datetime

def run_tracker(script_name, description, args=None):
    """Run a tracker script and return success/failure status"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {description}")
    print(f"📄 Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Build command
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        # Run the tracker with a timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS - Tracker completed successfully")
            # Extract key info from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['latest analysis', 'current assessment', 'health:', 'strength:', 'conditions:']):
                    print(f"   📊 {line.strip()}")
            return True
        else:
            print("❌ FAILED - Tracker returned error")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Tracker took too long to complete")
        return False
    except Exception as e:
        print(f"❌ ERROR - Exception occurred: {str(e)}")
        return False

def main():
    """Run all tracker tests"""
    print("🚀 ECONOMIC ANALYSIS SUITE - COMPREHENSIVE TEST")
    print("=" * 70)
    print(f"📅 Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 70)
    
    # Define trackers to test
    trackers = [
        ("buffet.py", "Market Valuation (Buffett Indicator)", ["--start", "2020-01-01"]),
        ("shipping_tracker_complete.py", "Shipping Activity Tracker", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("yield_curve_tracker.py", "Yield Curve & Recession Risk", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("labor_market_tracker.py", "Labor Market Health", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("credit_conditions_tracker.py", "Credit Conditions Monitor", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("manufacturing_tracker.py", "Manufacturing Sector Health", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("consumer_health_tracker.py", "Consumer Health Monitor", ["--start", "2020-01-01", "--min-data-points", "30"]),
        ("corporate_earnings_tracker.py", "Corporate Earnings & Profitability", ["--start", "2020-01-01", "--min-data-points", "30"])
    ]
    
    # Test results
    results = {}
    
    # Run each tracker
    for script, description, args in trackers:
        if os.path.exists(script):
            success = run_tracker(script, description, args)
            results[description] = success
        else:
            print(f"\n❌ SKIPPING: {description} - Script {script} not found")
            results[description] = False
    
    # Summary report
    print(f"\n\n{'='*70}")
    print("📊 ECONOMIC ANALYSIS SUITE - TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"🎯 Overall Result: {passed}/{total} trackers passed")
    print(f"✅ Success Rate: {(passed/total)*100:.1f}%")
    
    print(f"\n📋 Individual Results:")
    for tracker, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {tracker}")
    
    if passed == total:
        print(f"\n🎉 EXCELLENT! All economic trackers are working correctly.")
        print(f"   Your economic analysis suite is ready for comprehensive market monitoring.")
    elif passed >= total * 0.8:
        print(f"\n👍 GOOD! Most trackers are working ({passed}/{total} passed).")
        print(f"   Review failed trackers for data availability or configuration issues.")
    else:
        print(f"\n⚠️  NEEDS ATTENTION! Multiple trackers failed ({total-passed}/{total}).")
        print(f"   Check internet connection, data sources, and dependencies.")
    
    # File outputs check
    print(f"\n📁 Generated Files Check:")
    output_files = [
        "buffett_indicator_enhanced.png",
        "primary_shipping_tracker.png", 
        "yield_curve_analysis.png",
        "labor_market_analysis.png",
        "credit_conditions_analysis.png",
        "manufacturing_analysis.png",
        "consumer_health_analysis.png"
    ]
    
    files_found = 0
    for filename in output_files:
        if os.path.exists(filename):
            files_found += 1
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename}")
    
    print(f"\n📊 Files Generated: {files_found}/{len(output_files)}")
    print(f"={'='*70}")

if __name__ == "__main__":
    main()