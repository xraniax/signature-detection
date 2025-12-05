# src/convert_gif_to_png.py
from pathlib import Path
from PIL import Image
import os

def convert_gif_to_png():
    """Convert all GIF files to PNG"""
    print("Converting GIF to PNG...")
    
    directories = ["data/cedar_manual/genuine", "data/cedar_manual/forged"]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
            
        print(f"\nProcessing {dir_path}...")
        gif_files = list(Path(dir_path).glob("*.gif"))
        
        for i, gif_path in enumerate(gif_files):
            if i % 20 == 0:
                print(f"  Converted {i}/{len(gif_files)}...")
            
            # Open GIF and save as PNG
            img = Image.open(gif_path)
            png_path = gif_path.with_suffix('.png')
            img.save(png_path)
            
            # Remove original GIF (optional)
            # gif_path.unlink()
    
    print("\n✓ Conversion complete!")

if __name__ == "__main__":
    convert_gif_to_png()
