# src/download_cedar.py
import os
import zipfile
import tarfile
from pathlib import Path
import requests
import time

def download_cedar():
    """Download CEDAR signature dataset"""
    print("=" * 60)
    print("DOWNLOADING CEDAR SIGNATURE DATASET")
    print("=" * 60)
    
    # Create directory
    data_dir = Path("data/cedar_dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("CEDAR is the most widely used academic signature dataset.")
    print("It contains 55 people, each with:")
    print("  - 24 genuine signatures")
    print("  - 24 forged signatures")
    print("\nDownload options:")
    print("1. Official source: https://cedar.buffalo.edu/NIJ/data/signatures.rar")
    print("2. Alternative mirrors (if official fails)")
    
    # Try multiple sources
    sources = [
        {
            "name": "Academic Mirror 1",
            "url": "https://www.iapr-tc11.org/mediawiki/images/8/8a/CEDAR.zip",
            "type": "zip"
        },
        {
            "name": "Academic Mirror 2", 
            "url": "http://www.iapr-tc11.org/dataset/CEDAR/CEDAR.zip",
            "type": "zip"
        }
    ]
    
    for source in sources:
        print(f"\nTrying {source['name']}: {source['url']}")
        
        try:
            # Download
            response = requests.get(source['url'], stream=True, timeout=30)
            if response.status_code == 200:
                # Save file
                if source['type'] == 'zip':
                    zip_path = data_dir / "cedar.zip"
                else:
                    zip_path = data_dir / "cedar.rar"
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f"✓ Download successful: {zip_path}")
                
                # Extract
                print("Extracting...")
                if source['type'] == 'zip':
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print("✓ Extracted ZIP file")
                else:
                    # Would need unrar for .rar files
                    print("Note: RAR file - need unrar to extract")
                
                # Organize the data
                organize_cedar_data(data_dir)
                return True
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Automatic download failed. Please:")
    print("1. Go to: https://cedar.buffalo.edu/NIJ/data/signatures.rar")
    print("2. Download signatures.rar")
    print(f"3. Extract to: {data_dir}/")
    print("4. Run: python src/organize_cedar.py")
    print("\nThe dataset should have folders: full_org, full_forg")
    
    return False

def organize_cedar_data(data_dir):
    """Organize CEDAR dataset into our format"""
    print("\nOrganizing CEDAR dataset...")
    
    # CEDAR typically has folders: full_org (genuine), full_forg (forged)
    genuine_source = data_dir / "full_org"
    forged_source = data_dir / "full_forg"
    
    # Check different possible structures
    possible_genuine = ["full_org", "Original", "Genuine", "original"]
    possible_forged = ["full_forg", "Forged", "Forgery", "forged"]
    
    for gen_name in possible_genuine:
        genuine_source = data_dir / gen_name
        if genuine_source.exists():
            break
    
    for forg_name in possible_forged:
        forged_source = data_dir / forg_name
        if forged_source.exists():
            break
    
    if not genuine_source.exists() or not forged_source.exists():
        print("Could not find CEDAR folder structure.")
        print(f"Looking for: {data_dir}/full_org/ and {data_dir}/full_forg/")
        print(f"Found in {data_dir}:")
        for item in data_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return False
    
    # Create organized structure
    organized_dir = Path("data/cedar_organized")
    genuine_dir = organized_dir / "genuine"
    forged_dir = organized_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy genuine signatures
    print(f"Copying genuine signatures from {genuine_source}")
    genuine_files = list(genuine_source.glob("*.png")) + list(genuine_source.glob("*.gif"))
    
    for i, file_path in enumerate(genuine_files):
        if i % 50 == 0:
            print(f"  Processed {i} genuine signatures...")
        
        # Convert to PNG if GIF
        if file_path.suffix.lower() == '.gif':
            from PIL import Image
            img = Image.open(file_path)
            new_path = genuine_dir / f"genuine_{i:04d}.png"
            img.save(new_path)
        else:
            import shutil
            new_path = genuine_dir / f"genuine_{i:04d}.png"
            shutil.copy2(file_path, new_path)
    
    # Copy forged signatures
    print(f"\nCopying forged signatures from {forged_source}")
    forged_files = list(forged_source.glob("*.png")) + list(forged_source.glob("*.gif"))
    
    for i, file_path in enumerate(forged_files):
        if i % 50 == 0:
            print(f"  Processed {i} forged signatures...")
        
        if file_path.suffix.lower() == '.gif':
            from PIL import Image
            img = Image.open(file_path)
            new_path = forged_dir / f"forged_{i:04d}.png"
            img.save(new_path)
        else:
            import shutil
            new_path = forged_dir / f"forged_{i:04d}.png"
            shutil.copy2(file_path, new_path)
    
    print(f"\n✓ CEDAR dataset organized:")
    print(f"  Genuine: {len(list(genuine_dir.glob('*.png')))} signatures")
    print(f"  Forged: {len(list(forged_dir.glob('*.png')))} signatures")
    print(f"  Location: {organized_dir}")
    
    return True

if __name__ == "__main__":
    download_cedar()