# src/test_step2.py
print("=== STEP 2: Testing Siamese Model ===")

try:
    # Try to import the model
    from siamese_resnet import SiameseResNet
    print("✓ Successfully imported SiameseResNet")
    
    # Create a simple model
    model = SiameseResNet(embedding_dim=128)
    print(f"✓ Model created")
    print(f"  Embedding dimension: 128")
    
    # Test with dummy input
    import torch
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"✓ Created dummy input: shape {dummy_input.shape}")
    
    # Test forward pass for one image
    with torch.no_grad():
        embedding = model.forward_one(dummy_input)
        print(f"✓ Forward pass works")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Expected: torch.Size([{batch_size}, 128])")
        
        # Test Siamese forward (two images)
        dummy_input2 = torch.randn(batch_size, 3, 224, 224)
        embedding1, embedding2 = model(dummy_input, dummy_input2)
        print(f"✓ Siamese forward works")
        print(f"  Embedding 1 shape: {embedding1.shape}")
        print(f"  Embedding 2 shape: {embedding2.shape}")
        
        # Test distance calculation
        distance = model.get_distance(embedding1, embedding2)
        print(f"✓ Distance calculation works")
        print(f"  Distance shape: {distance.shape}")
        print(f"  Sample distances: {distance[0].item():.4f}, {distance[1].item():.4f}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nLet's check the siamese_resnet.py file...")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()