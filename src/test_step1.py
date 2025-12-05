# src/test_step1.py
print("=== STEP 1: Testing siamese_dataset.py ===")

try:
    # Try to import
    from siamese_dataset import SiameseSignatureDataset
    print("✓ Successfully imported SiameseSignatureDataset")
    
    # Check if we have test data
    import os
    if os.path.exists("data/simple_verification"):
        print("✓ Test data directory exists")
        
        # Try to create dataset
        try:
            dataset = SiameseSignatureDataset("data/simple_verification")
            print(f"✓ Dataset created with {len(dataset)} pairs")
            
            # Try to get one item
            img1, img2, label = dataset[0]
            print(f"✓ Got sample: image shapes {img1.shape}, {img2.shape}, label {label}")
            
        except Exception as e:
            print(f"✗ Error creating dataset: {e}")
            
    else:
        print("✗ Test data not found at data/simple_verification/")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Let's check what's in the file...")
    
    # Read the file
    with open("src/siamese_dataset.py", "r") as f:
        lines = f.readlines()
        print(f"\nFile has {len(lines)} lines")
        print("First 5 lines:")
        for i in range(min(5, len(lines))):
            print(f"  {i+1}: {lines[i].strip()}")