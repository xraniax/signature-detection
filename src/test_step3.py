# src/test_step3.py
print("=== STEP 3: Testing Loss Function ===")

try:
    from siamese_resnet import ContrastiveLoss
    print("✓ Successfully imported ContrastiveLoss")
    
    import torch
    
    # Create dummy embeddings and labels
    embedding1 = torch.randn(4, 128)  # 4 samples, 128-dim embeddings
    embedding2 = torch.randn(4, 128)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Mix of genuine/forged
    
    print(f"✓ Created test data:")
    print(f"  Embedding 1 shape: {embedding1.shape}")
    print(f"  Embedding 2 shape: {embedding2.shape}")
    print(f"  Labels: {labels}")
    
    # Test loss
    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(embedding1, embedding2, labels)
    
    print(f"✓ Loss calculation works")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test with different margin
    criterion2 = ContrastiveLoss(margin=2.0)
    loss2 = criterion2(embedding1, embedding2, labels)
    print(f"  Loss with margin=2.0: {loss2.item():.4f}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()