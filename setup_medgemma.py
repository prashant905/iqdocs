#!/usr/bin/env python3
"""
Setup script for MedGemma authentication and license acceptance.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📋 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Command: {command}")
        print(f"Error output: {e.stderr}")
        return False

def check_huggingface_cli():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed")
        return True
    except ImportError:
        print("❌ huggingface_hub is not installed")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    commands = [
        "pip install huggingface_hub>=0.25.0",
        "pip install transformers>=4.50.0",
        "pip install accelerate>=0.20.0"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Installing {cmd.split()[-1]}"):
            return False
    return True

def login_to_huggingface():
    """Guide user through Hugging Face login."""
    print("\n🔐 Hugging Face Authentication")
    print("="*50)
    print("You need to login to Hugging Face to access MedGemma.")
    print("Please follow these steps:")
    print("\n1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token")
    print("4. Run the login command below")
    
    print("\n💻 Run this command in your terminal:")
    print("huggingface-cli login")
    
    input("\nPress Enter after you've completed the login...")

def accept_license():
    """Guide user to accept MedGemma license."""
    print("\n📋 License Agreement")
    print("="*50)
    print("You need to accept the MedGemma license to use the model.")
    print("\n🌐 Please visit: https://huggingface.co/google/medgemma-4b-it")
    print("👆 Click 'Agree and access repository' after reviewing the terms")
    
    input("\nPress Enter after you've accepted the license...")

def test_model_access():
    """Test if MedGemma model can be accessed."""
    print("\n🧪 Testing Model Access")
    print("="*50)
    
    test_code = """
try:
    from transformers import AutoTokenizer
    print("📥 Testing MedGemma access...")
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    print("✅ MedGemma model access successful!")
    print("🎉 Setup complete! You can now use MedGemma in the app.")
except Exception as e:
    print(f"❌ Error accessing MedGemma: {e}")
    print("Please ensure you've completed the authentication and license steps.")
"""
    
    exec(test_code)

def main():
    """Main setup function."""
    print("🏥 MedGemma Setup Assistant")
    print("="*50)
    print("This script will help you set up MedGemma for local use.")
    print("MedGemma is Google's medical specialist AI model.")
    
    # Step 1: Check and install requirements
    if not check_huggingface_cli():
        print("\n📦 Installing required packages...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("pip install huggingface_hub transformers accelerate")
            return False
    
    # Step 2: Guide through login
    login_to_huggingface()
    
    # Step 3: Guide through license acceptance
    accept_license()
    
    # Step 4: Test access
    test_model_access()
    
    print("\n🎊 Setup Process Complete!")
    print("You can now run the medical analysis app with MedGemma support:")
    print("python app.py")

if __name__ == "__main__":
    main() 