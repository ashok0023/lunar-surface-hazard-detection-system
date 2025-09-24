#!/usr/bin/env python3
"""
Simple launcher for the Lunar Hazard Detection Web Application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import cv2
        import ultralytics
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r web_app_requirements.txt")
        return False

def check_model():
    """Check if YOLO model exists"""
    model_path = Path('yolov8n.pt')
    if model_path.exists():
        print("✅ YOLOv8 model found!")
        return True
    else:
        print("⚠️  YOLOv8 model not found. The app will try to download it automatically.")
        return True  # Continue anyway

def check_sample_data():
    """Check if sample lunar images exist"""
    sample_dir = Path('data/augmented')
    if sample_dir.exists() and len(list(sample_dir.glob('*.jpg'))) > 0:
        print(f"✅ Sample lunar images found: {len(list(sample_dir.glob('*.jpg')))} images")
        return True
    else:
        print("⚠️  No sample lunar images found. You can still upload your own images.")
        return True  # Continue anyway

def main():
    """Main launcher function"""
    print("🌙 Lunar Hazard Detection Web Application Launcher")
    print("=" * 50)

    # Check system requirements
    print("\n🔍 Checking requirements...")

    all_good = True
    all_good &= check_requirements()
    all_good &= check_model()
    all_good &= check_sample_data()

    if not all_good:
        print("\n❌ Please fix the issues above before running the application.")
        sys.exit(1)

    print("\n🚀 Starting Lunar Hazard Detection Web Application...")
    print("📱 Open your browser and navigate to: http://localhost:8080")
    print("⏹️  Press Ctrl+C to stop the application")

    try:
        # Start the Flask application
        import web_app
        web_app.app.run(debug=True, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Make sure no other application is using port 5000.")
        sys.exit(1)

if __name__ == "__main__":
    main()
