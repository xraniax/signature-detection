# src/create_test_verification_data.py
from pathlib import Path
from PIL import Image
import numpy as np
import os

def create_test_dataset():
    """Create a test verification dataset"""
    print("Creating test verification dataset...")
    
    base_dir = Path("data/test_verification")
    genuine_dir = base_dir / "genuine"
    forged_dir = base_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 genuine signatures
    print("Creating genuine signatures...")
    for i in range(10):
        # Create a simple signature-like image
        img = Image.new('RGB', (300, 150), color='white')
        # Draw a simple "signature" line
        img_np = np.array(img)
        
        # Draw a curved line (simulating signature)
        for x in range(50, 250):
            y = 75 + int(20 * np.sin(x / 20))
            if 0 <= y < 150:
                img_np[y-2:y+2, x-2:x+2] = [0, 0, 0]  # Black pixels
        
        img = Image.fromarray(img_np)
        img.save(genuine_dir / f"genuine_{i:03d}.png")
    
    # Create 10 forged signatures (variations of genuine ones)
    print("Creating forged signatures...")
    for i in range(10):
        # Load a genuine signature
        genuine_path = genuine_dir / f"genuine_{i%5:03d}.png"
        img = Image.open(genuine_path)
        
        # Apply transformations to simulate forgery
        if i % 3 == 0:
            img = img.rotate(5)  # Rotated
        elif i % 3 == 1:
            img = img.resize((280, 140))  # Scaled
        else:
            # Add noise
            img_np = np.array(img)
            noise = np.random.randint(-30, 30, img_np.shape, dtype=np.int32)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        
        img.save(forged_dir / f"forged_{i:03d}.png")
    
    print(f"Dataset created at: {base_dir}")
    print(f"Genuine signatures: {len(list(genuine_dir.glob('*.png')))}")
    print(f"Forged signatures: {len(list(forged_dir.glob('*.png')))}")
    
    return base_dir

if __name__ == "__main__":
    create_test_dataset()