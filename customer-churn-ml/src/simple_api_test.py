#!/usr/bin/env python
"""
Simple API Test - Check imports and basic functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("🔧 Testing API imports...")

try:
    print("   Testing config import...")
    from config import PROJECT_ROOT
    print(f"   ✅ Config imported - PROJECT_ROOT: {PROJECT_ROOT}")
    
    print("   Testing validation import...")
    from validation import validate_churn_data
    print("   ✅ Validation imported")
    
    print("   Testing predict import...")
    from predict import ChurnPredictor
    print("   ✅ Predict imported")
    
    print("\n🚀 Starting minimal API server...")
    
    # Create minimal FastAPI app
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="Test API")
    
    @app.get("/")
    def read_root():
        return {"message": "API test successful"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "timestamp": "2026-02-09"}
    
    print("✅ All imports successful!")
    print("\n🌐 Starting server on http://localhost:8000")
    print("   - Health check: http://localhost:8000/health")
    print("   - API docs: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop the server")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying to diagnose...")
    
    # Check what files exist
    src_dir = Path(__file__).parent
    print(f"\nFiles in {src_dir}:")
    for file in src_dir.glob("*.py"):
        print(f"   - {file.name}")

except Exception as e:
    print(f"❌ Error: {e}")
