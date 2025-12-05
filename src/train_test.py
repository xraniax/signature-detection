# src/train_test.py
import torch
from torch.utils.data import DataLoader
from siamese_dataset import SiameseSignatureDataset
from siamese_resnet import SiameseResNet, ContrastiveLoss
import torch.optim as optim
from pathlib import Path

print("=" * 60)
print("TRAINING TEST WITH AVAILABLE DATA")
print("=" * 60)

# Check what data we have
data_dirs = [
    "data/real_signatures",
    "data/cedar_dataset", 
    "data/simple_verification",
    "data/test_verification"
]

available_data = []
for dir_path in data_dirs:
    if Path(dir_path).exists():
        genuine = len(list(Path(dir_path).glob("genuine/*.png")))
        forged = len(list(Path(dir_path).glob("forged/*.png")))
        if genuine > 0 and forged > 0:
            available_data.append((dir_path, genuine + forged))

if not available_data:
    print("❌ No training data found!")
    print("\nPlease:")
    print("1. Run: python src/create_simple_test_data.py")
    print("2. Or get real data (CEDAR dataset)")
    exit()

print("\nAvailable datasets:")
for dir_path, count in available_data:
    print(f"  {dir_path}: {count} images")

# Use the first available dataset
data_dir = available_data[0][0]
print(f"\nUsing: {data_dir}")

# Configuration
config = {
    'batch_size': 8,
    'learning_rate': 0.0001,
    'epochs': 10,
    'embedding_dim': 128
}

print(f"\nConfiguration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Create dataset and dataloader
print("\nCreating dataset...")
dataset = SiameseSignatureDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
print(f"Dataset: {len(dataset)} pairs")
print(f"Batches: {len(dataloader)}")

# Create model
print("\nCreating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseResNet(embedding_dim=config['embedding_dim']).to(device)
print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
print(f"\nStarting training for {config['epochs']} epochs...")
print("-" * 60)

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        embedding1, embedding2 = model(img1, img2)
        loss = criterion(embedding1, embedding2, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            distances = torch.norm(embedding1 - embedding2, dim=1)
            predictions = (distances < 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
          f"Loss: {avg_loss:.4f} | "
          f"Accuracy: {accuracy:.1f}%")

print("-" * 60)

# Save model
save_dir = Path("models")
save_dir.mkdir(exist_ok=True)
model_path = save_dir / "siamese_trained_test.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")

# Test the model
print("\nTesting model...")
model.eval()
with torch.no_grad():
    # Get a test sample
    test_img1, test_img2, test_label = dataset[0]
    test_img1 = test_img1.unsqueeze(0).to(device)
    test_img2 = test_img2.unsqueeze(0).to(device)
    
    emb1, emb2 = model(test_img1, test_img2)
    distance = torch.norm(emb1 - emb2).item()
    
    print(f"Test sample:")
    print(f"  True label: {test_label.item()} (0=forged, 1=genuine)")
    print(f"  Distance: {distance:.4f}")
    print(f"  Prediction: {'GENUINE' if distance < 0.5 else 'FORGED'}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nNext: Get real CEDAR dataset for better results.")
print("URL: https://cedar.buffalo.edu/NIJ/data/signatures.rar")