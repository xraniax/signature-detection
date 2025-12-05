# src/setup_cedar_manual.py
from pathlib import Path

def setup_cedar_manual():
    """Manual setup instructions for CEDAR"""
    print("=" * 60)
    print("MANUAL CEDAR DATASET SETUP")
    print("=" * 60)
    
    # Create directory structure
    data_dir = Path("data/cedar_manual")
    genuine_dir = data_dir / "genuine"
    forged_dir = data_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {data_dir}")
    print("\nINSTRUCTIONS:")
    print("1. Download CEDAR dataset from:")
    print("   https://cedar.buffalo.edu/NIJ/data/signatures.rar")
    print("\n2. Extract the RAR file")
    print("   You should get folders: 'full_org' and 'full_forg'")
    print("\n3. Copy all images from:")
    print(f"   - full_org/*.gif  →  {genuine_dir}/")
    print(f"   - full_forg/*.gif →  {forged_dir}/")
    print("\n4. Convert GIF to PNG (optional - our loader handles GIF):")
    print("   python src/convert_gif_to_png.py")
    print("\n5. Run training:")
    print("   python src/train_siamese_complete.py")
    
    # Create a simple script to convert GIF to PNG
    create_conversion_script()
    
    return data_dir

def create_conversion_script():
    """Create GIF to PNG conversion script"""
    script = """# src/convert_gif_to_png.py
from pathlib import Path
from PIL import Image
import os

def convert_gif_to_png():
    \"\"\"Convert all GIF files to PNG\"\"\"
    print("Converting GIF to PNG...")
    
    directories = ["data/cedar_manual/genuine", "data/cedar_manual/forged"]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
            
        print(f"\\nProcessing {dir_path}...")
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
    
    print("\\n✓ Conversion complete!")

if __name__ == "__main__":
    convert_gif_to_png()
"""
    
    with open("src/convert_gif_to_png.py", "w") as f:
        f.write(script)
    
    print(f"\nCreated conversion script: src/convert_gif_to_png.py")

if __name__ == "__main__":
    setup_cedar_manual()