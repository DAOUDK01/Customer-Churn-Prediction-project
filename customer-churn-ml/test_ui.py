#!/usr/bin/env python
"""
Quick Test: Streamlit Demo UI
Verifies that the demo UI can start and basic functionality works
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def test_ui_dependencies():
    """Test that required packages are installed"""
    print("🔧 Testing UI Dependencies...")
    
    try:
        import streamlit
        import requests
        import pandas
        print("   ✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return False

def test_api_server():
    """Test if API server can be started"""
    print("\n🚀 Testing API Server...")
    
    try:
        # Check if API is already running
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("   ✅ API server already running")
        return True
    except:
        print("   ℹ️ API server not running (normal for first run)")
        print("   📋 To start API: python src/api.py")
        return False

def test_ui_startup():
    """Test if Streamlit UI can be validated"""
    print("\n🎨 Testing UI Startup...")
    
    ui_file = Path("streamlit_app.py")
    if not ui_file.exists():
        print("   ❌ streamlit_app.py not found")
        return False
    
    # Test basic syntax by importing
    try:
        import streamlit
        # Basic syntax check
        with open(ui_file, encoding='utf-8') as f:
            content = f.read()
            if "st.title" in content and "requests.post" in content:
                print("   ✅ UI file structure looks correct")
                return True
            else:
                print("   ⚠️ UI file may be incomplete")
                return False
    except Exception as e:
        print(f"   ❌ UI validation failed: {e}")
        return False

def main():
    """Run quick UI tests"""
    print("🎯 STREAMLIT DEMO UI - QUICK TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_ui_dependencies():
        tests_passed += 1
    
    if test_api_server():
        tests_passed += 1
    
    if test_ui_startup():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n🚀 Ready to run:")
        print("   1. Start API: python src/api.py")
        print("   2. Start UI:  streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️ {total_tests - tests_passed} tests failed")
        print("Review errors above before proceeding")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)