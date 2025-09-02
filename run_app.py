"""
Music Popularity Predictor - Streamlit App Launcher
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def run_streamlit():
    """Launch the Streamlit app"""
    print("🚀 Launching Music Popularity Predictor...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("🎵 MUSIC POPULARITY PREDICTOR")
    print("="*50)
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    try:
        # Check if requirements are installed
        import streamlit
        import plotly
        print("✅ All requirements are already installed!")
    except ImportError:
        print("📦 Installing missing requirements...")
        install_requirements()
    
    # Launch the app
    run_streamlit()