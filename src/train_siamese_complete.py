# src/train_siamese_complete.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our modules
from siamese_resnet import SiameseResNet, ContrastiveLoss, TripletLoss
from siamese_dataset import SiameseSignatureDataset, TripletSignatureDataset

class SiameseTrainer:
    """Complete trainer for Siamese signature verification"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directories
        self.create_directories()
        
        # Initialize model
        self.model = self.init_model()
        
        # Loss and optimizer
        self.criterion, self.optimizer = self.init_training()
        
        # Data loaders
        self.train_loader, self.val_loader = self.init_dataloaders()
    
    def create_directories(self):
        """Create necessary directories"""
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
    
    def init_model(self):
        """Initialize Siamese model"""
        model = SiameseResNet(
            embedding_dim=self.config['embedding_dim'],
            pretrained=self.config['pretrained']
        ).to(self.device)
        
        print(f"Model initialized with {self.count_parameters(model):,} parameters")
        return model
    
    def count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def init_training(self):
        """Initialize loss function and optimizer"""
        if self.config['loss_type'] == 'contrastive':
            criterion = ContrastiveLoss(margin=self.config['margin'])
        elif self.config['loss_type'] == 'triplet':
            criterion = TripletLoss(margin=self.config['margin'])
        else:
            raise ValueError(f"Unknown loss type: {self.config['loss_type']}")
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return criterion, optimizer
    
    def init_dataloaders(self):
        """Initialize data loaders"""
        if self.config['loss_type'] == 'triplet':
            train_dataset = TripletSignatureDataset(
                self.config['train_data_dir']
            )
            val_dataset = TripletSignatureDataset(
                self.config['val_data_dir']
            ) if self.config['val_data_dir'] else None
        else:
            train_dataset = SiameseSignatureDataset(
                self.config['train_data_dir'],
                mode='train',
                pair_per_sample=self.config['pairs_per_sample']
            )
            val_dataset = SiameseSignatureDataset(
                self.config['val_data_dir'],
                mode='val',
                pair_per_sample=self.config['pairs_per_sample']
            ) if self.config['val_data_dir'] else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers']
            )
        
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            if self.config['loss_type'] == 'triplet':
                anchor, positive, negative = batch_data
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                
                # Forward pass
                anchor_emb = self.model.forward_one(anchor)
                positive_emb = self.model.forward_one(positive)
                negative_emb = self.model.forward_one(negative)
                
                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            else:
                img1, img2, labels = batch_data
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                # Forward pass
                embedding1, embedding2 = self.model(img1, img2)
                loss = self.criterion(embedding1, embedding2, labels)
                
                # Compute accuracy
                distances = self.model.get_distance(embedding1, embedding2)
                predictions = (distances < self.config['threshold']).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            if self.config['loss_type'] == 'contrastive':
                accuracy = 100. * correct / total if total > 0 else 0
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%'})
            else:
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        if not self.val_loader:
            return None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                if self.config['loss_type'] == 'triplet':
                    anchor, positive, negative = batch_data
                    anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                    
                    anchor_emb = self.model.forward_one(anchor)
                    positive_emb = self.model.forward_one(positive)
                    negative_emb = self.model.forward_one(negative)
                    
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                else:
                    img1, img2, labels = batch_data
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    
                    embedding1, embedding2 = self.model(img1, img2)
                    loss = self.criterion(embedding1, embedding2, labels)
                    
                    distances = self.model.get_distance(embedding1, embedding2)
                    predictions = (distances < self.config['threshold']).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        if self.config['loss_type'] == 'contrastive':
            accuracy = 100. * correct / total if total > 0 else 0
            return avg_loss, accuracy
        else:
            return avg_loss, None
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training configuration: {self.config}")
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader:
                val_results = self.validate()
                if val_results:
                    val_loss, val_accuracy = val_results
                    val_losses.append(val_loss)
                    
                    if val_accuracy is not None:
                        val_accuracies.append(val_accuracy)
                        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                    else:
                        print(f"Val Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(f"best_model.pth")
                        print(f"Best model saved with val loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")
        
        # Save final model
        self.save_model("final_model.pth")
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, val_accuracies)
        
        print("Training completed!")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.config['epochs'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, f"models/{filename}")
        print(f"Model saved to: models/{filename}")
    
    def plot_training_history(self, train_losses, val_losses, val_accuracies):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy if available
        if val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
        plt.show()

# Configuration
CONFIG = {
    # Model
    'embedding_dim': 128,
    'pretrained': True,
    
    # Training
    'loss_type': 'contrastive',  # 'contrastive' or 'triplet'
    'margin': 1.0,
    'threshold': 0.5,
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'epochs': 50,
    'batch_size': 32,
    'pairs_per_sample': 3,
    
    # Data
    'train_data_dir': 'data/cedar_manual',
    'val_data_dir': None,  # Set to validation directory if available
    'num_workers': 4,
    
    # Checkpointing
    'save_interval': 10
}

def main():
    """Main training function"""
    print("Siamese Signature Verification Training")
    print("=" * 50)
    
    # Create trainer
    trainer = SiameseTrainer(CONFIG)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()