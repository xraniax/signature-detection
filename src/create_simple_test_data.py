# src/create_simple_test_data.py
from pathlib import Path
from PIL import Image
import numpy as np

def create_simple_test_data():
    """Create simple test data"""
    print("Creating simple test verification dataset...")
    
    # Create directory
    base_dir = Path("data/simple_verification")
    genuine_dir = base_dir / "genuine"
    forged_dir = base_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some images
    for i in range(5):
        # Genuine signature
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        img.save(genuine_dir / f"gen_{i}.png")
        print(f"Created: {genuine_dir}/gen_{i}.png")
        
        # Forged signature (slightly different)
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        img.save(forged_dir / f"forged_{i}.png")
        print(f"Created: {forged_dir}/forged_{i}.png")
    
    print(f"\nDataset created at: {base_dir}")
    print(f"Total images: 10")
    return True

if __name__ == "__main__":
    success = create_simple_test_data()
    if success:
        print("SUCCESS: Test data created!")
    else:
        print("FAILED: Could not create test data")