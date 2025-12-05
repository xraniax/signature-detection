# src/test_step5.py
print("=== STEP 5: Testing Complete Training Script ===")

# Check if we can import the complete trainer
try:
    from train_siamese_complete import CONFIG, SiameseTrainer
    print("✓ Successfully imported training modules")
    
    # Modify config for testing
    test_config = CONFIG.copy()
    test_config.update({
        'epochs': 2,  # Just 2 epochs for testing
        'batch_size': 4,
        'train_data_dir': 'data/simple_verification',
        'val_data_dir': None,  # No validation for test
    })
    
    print(f"\nTest configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    
    # Try to create trainer (but don't actually train)
    print("\nCreating trainer (testing initialization only)...")
    try:
        trainer = SiameseTrainer(test_config)
        print("✓ Trainer created successfully")
        print("\nThe complete training system is ready!")
        print("\nTo actually train, run:")
        print("python src/train_siamese_complete.py")
        print("\nOr with custom config:")
        print("""
# In your own script:
from train_siamese_complete import SiameseTrainer

config = {
    'embedding_dim': 128,
    'epochs': 50,
    'batch_size': 32,
    'train_data_dir': 'data/your_dataset',
    # ... other settings
}

trainer = SiameseTrainer(config)
trainer.train()
        """)
        
    except Exception as e:
        print(f"✗ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nLet's check the train_siamese_complete.py file...")
    
    # Check if file exists
    import os
    if os.path.exists("src/train_siamese_complete.py"):
        print("File exists. Checking content...")
        with open("src/train_siamese_complete.py", "r") as f:
            lines = f.readlines()
            print(f"File has {len(lines)} lines")
    else:
        print("File doesn't exist.")