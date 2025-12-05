# src/test_step4.py
print("=== STEP 4: Testing Training Pipeline ===")

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Import our working components
from siamese_dataset import SiameseSignatureDataset
from siamese_resnet import SiameseResNet, ContrastiveLoss

# Configuration
config = {
    'batch_size': 4,
    'learning_rate': 0.0001,
    'epochs': 3  # Just 3 epochs for testing
}

print(f"Configuration: {config}")

# 1. Create dataset and dataloader
print("\n1. Creating dataset and dataloader...")
dataset = SiameseSignatureDataset("data/simple_verification")
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print(f"   ✓ Dataset: {len(dataset)} pairs")
print(f"   ✓ Dataloader: {len(dataloader)} batches")

# 2. Create model
print("\n2. Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

model = SiameseResNet(embedding_dim=128).to(device)
print(f"   ✓ Model created on {device}")

# 3. Create loss and optimizer
print("\n3. Setting up loss and optimizer...")
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
print(f"   ✓ Contrastive loss (margin=1.0)")
print(f"   ✓ Adam optimizer (lr={config['learning_rate']})")

# 4. Test one training batch
print("\n4. Testing one training batch...")
model.train()  # Set to training mode

for batch_idx, (img1, img2, labels) in enumerate(dataloader):
    if batch_idx >= 1:  # Just test first batch
        break
    
    # Move to device
    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
    
    print(f"   Batch {batch_idx}:")
    print(f"     Images shape: {img1.shape}")
    print(f"     Labels: {labels.cpu().numpy()}")
    
    # Forward pass
    optimizer.zero_grad()
    embedding1, embedding2 = model(img1, img2)
    
    # Compute loss
    loss = criterion(embedding1, embedding2, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"     Loss: {loss.item():.4f}")
    print(f"     Embeddings shape: {embedding1.shape}")
    
    # Compute accuracy
    with torch.no_grad():
        distances = torch.norm(embedding1 - embedding2, dim=1)
        predictions = (distances < 0.5).float()  # Simple threshold
        accuracy = (predictions == labels).float().mean().item() * 100
        print(f"     Accuracy: {accuracy:.1f}%")

# 5. Test evaluation mode
print("\n5. Testing evaluation mode...")
model.eval()  # Set to evaluation mode

with torch.no_grad():
    test_img1, test_img2, test_label = dataset[0]
    test_img1 = test_img1.unsqueeze(0).to(device)  # Add batch dimension
    test_img2 = test_img2.unsqueeze(0).to(device)
    
    emb1, emb2 = model(test_img1, test_img2)
    distance = torch.norm(emb1 - emb2).item()
    
    print(f"   Single sample test:")
    print(f"     Label: {test_label.item()} (0=forged, 1=genuine)")
    print(f"     Distance: {distance:.4f}")
    print(f"     Prediction: {'GENUINE' if distance < 0.5 else 'FORGED'}")

print("\n" + "="*50)
print("✅ TRAINING PIPELINE TEST COMPLETE!")
print("\nNext: You can run full training with:")
print("python src/train_siamese_complete.py")
print("\nOr test with real data when ready.")