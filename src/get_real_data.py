# src/get_real_data.py
from pathlib import Path
import os

print("=" * 60)
print("GETTING REAL SIGNATURE DATA FOR TRAINING")
print("=" * 60)

print("\nOPTION 1: Download CEDAR (Academic Standard)")
print("  URL: https://cedar.buffalo.edu/NIJ/data/signatures.rar")
print("  Size: ~50MB")
print("  Contains: 55 people × 24 genuine + 24 forged each")

print("\nOPTION 2: Use Kaggle Dataset (Easier)")
print("  Search: 'signature verification dataset' on Kaggle")
print("  Many public datasets available")
print("  Requires Kaggle account")

print("\nOPTION 3: Create from Your Detections")
print("  Use your working YOLO detector")
print("  Extract signatures from documents")
print("  Manually label as genuine/forged")

print("\n" + "=" * 60)
print("RECOMMENDED: OPTION 1 (CEDAR)")
print("=" * 60)

# Create directory
data_dir = Path("data/real_signatures")
genuine_dir = data_dir / "genuine"
forged_dir = data_dir / "forged"

genuine_dir.mkdir(parents=True, exist_ok=True)
forged_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDirectory structure created:")
print(f"  {data_dir}/")
print(f"  ├── genuine/  (place genuine signatures here)")
print(f"  └── forged/   (place forged signatures here)")

print("\n" + "=" * 60)
print("QUICK START: Use Test Data for Now")
print("=" * 60)

# Copy our test data as placeholder
import shutil
test_dir = Path("data/simple_verification")

if test_dir.exists():
    print(f"\nCopying test data to {data_dir}/...")
    
    # Copy genuine
    for img in (test_dir / "genuine").glob("*.png"):
        shutil.copy2(img, genuine_dir / img.name)
    
    # Copy forged
    for img in (test_dir / "forged").glob("*.png"):
        shutil.copy2(img, forged_dir / img.name)
    
    print(f"Copied 10 test signatures to {data_dir}/")
    print("You can train with this test data while getting real data.")
else:
    print("\nTest data not found. Creating minimal test set...")
    
    from PIL import Image
    # Create 5 genuine
    for i in range(5):
        img = Image.new('RGB', (224, 224), color='white')
        img.save(genuine_dir / f"genuine_{i}.png")
    
    # Create 5 forged
    for i in range(5):
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        img.save(forged_dir / f"forged_{i}.png")
    
    print(f"Created 10 test signatures in {data_dir}/")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Get real signature dataset (CEDAR recommended)")
print(f"2. Place in: {data_dir}/")
print("3. Run training: python src/train_siamese_complete.py")
print("\nFor now, you can train with the test data:")
print("  python src/train_test.py")