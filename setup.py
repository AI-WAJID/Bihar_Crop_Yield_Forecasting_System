
import subprocess
import sys
import os

def main():
    print("🚀 Setting up Bihar Crop Forecasting System...")

    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the system, run:")
    print("    python run_project.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
