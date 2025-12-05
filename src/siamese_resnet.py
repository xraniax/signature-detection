# src/siamese_resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class SiameseResNet(nn.Module):
    """Siamese Network with ResNet-18 backbone for signature verification"""
    
    def __init__(self, embedding_dim=128, pretrained=True):
        #Creates a twin network that processes two images simultaneously
        
        super(SiameseResNet, self).__init__()
        
        # Load ResNet-18 backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze early layers (optional)
        for param in list(self.feature_extractor.parameters())[:-4]:  # Unfreeze last 4 layers
            param.requires_grad = False
        
        # Add custom embedding layer
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
        # Distance metric
        self.distance = nn.PairwiseDistance(p=2)
    
    def forward_one(self, x):
        """Forward pass for one input"""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.fc(features)
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, input1, input2):
        """Forward pass for pair of inputs"""
        embedding1 = self.forward_one(input1)
        embedding2 = self.forward_one(input2)
        return embedding1, embedding2
    
    def get_distance(self, embedding1, embedding2):
        """Compute Euclidean distance between embeddings"""
        return self.distance(embedding1, embedding2)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1, embedding2: Embeddings from Siamese network
            label: 1 for genuine pair (same person), 0 for forged pair (different person)
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss formula
        loss_genuine = label * torch.pow(euclidean_distance, 2)
        loss_forged = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(loss_genuine + loss_forged)
        return loss

class TripletLoss(nn.Module):
    """Triplet loss for better embedding learning (alternative to contrastive)"""
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor signature embedding
            positive: Positive (genuine) signature embedding
            negative: Negative (forged) signature embedding
        """
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return torch.mean(losses)

# Image transformations for signatures
def get_signature_transforms(img_size=(224, 224)):
    """Get image transformations for signature preprocessing"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def test_model():
    """Test the Siamese network"""
    print("Testing SiameseResNet model...")
    
    # Create dummy input
    batch_size = 4
    dummy_img1 = torch.randn(batch_size, 3, 224, 224)
    dummy_img2 = torch.randn(batch_size, 3, 224, 224)
    
    # Create model
    model = SiameseResNet(embedding_dim=128)
    
    # Forward pass
    embedding1, embedding2 = model(dummy_img1, dummy_img2)
    
    print(f"Input shape: {dummy_img1.shape}")
    print(f"Embedding shape: {embedding1.shape}")
    print(f"Model test successful!")
    
    # Test loss
    criterion = ContrastiveLoss(margin=1.0)
    dummy_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Mix of genuine/forged
    loss = criterion(embedding1, embedding2, dummy_labels)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    model = test_model()