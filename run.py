#!/usr/bin/env python3
"""
StegoWave - Audio Steganography with AI Compression
Launcher script

Usage:
    python run.py

Then open frontend/index.html in your browser.
"""

import subprocess
import sys
import os

# Reconfigure console stream to UTF-8 to support emoji characters on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def check_deps():
    """Check if requirements are installed."""
    try:
        import flask, numpy, PIL, Crypto
        print("✅ All dependencies found.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r backend/requirements.txt")
        return False

if __name__ == '__main__':
    print("=" * 55)
    print("  🎵 StegoWave — Audio Steganography")
    print("=" * 55)

    if not check_deps():
        print("\nInstall dependencies first:")
        print("  cd backend")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("\n📡 Starting API server on http://localhost:5000")
    print("🌐 Open frontend/index.html in your browser")
    print("   (Or use Live Server in VS Code)\n")
    print("-" * 55)

    os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
    subprocess.run([sys.executable, 'app.py'])
