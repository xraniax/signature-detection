# src/siamese_dataset.py
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from pathlib import Path

class SiameseSignatureDataset(Dataset):
    """Dataset for Siamese network training with genuine/forged pairs"""
    
    def __init__(self, root_dir, transform=None, mode='train', pair_per_sample=5):
        """
        Args:
            root_dir: Directory with 'genuine' and 'forged' subdirectories
            transform: Image transformations
            mode: 'train' or 'test'
            pair_per_sample: Number of pairs to generate per genuine sample
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.pair_per_sample = pair_per_sample
        
        # Load all images
        self.genuine_images = self._load_images('genuine')
        self.forged_images = self._load_images('forged')
        
        print(f"Loaded {len(self.genuine_images)} genuine signatures")
        print(f"Loaded {len(self.forged_images)} forged signatures")
        
        # Create pairs
        self.pairs = self._create_pairs()
        print(f"Created {len(self.pairs)} training pairs")
        
        # Transformations
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_images(self, folder):
        """Load all images from a folder"""
        folder_path = self.root_dir / folder
        if not folder_path.exists():
            return []
        
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            images.extend(folder_path.glob(ext))
        
        return images
    
    def _create_pairs(self):
        """Create training pairs (genuine and forged)"""
        pairs = []
        
        # Create genuine pairs (same person signatures)
        for genuine_img in self.genuine_images:
            # Pair with other genuine signatures (same class)
            other_genuine = [img for img in self.genuine_images if img != genuine_img]
            if other_genuine:
                for _ in range(min(self.pair_per_sample, len(other_genuine))):
                    pair_img = random.choice(other_genuine)
                    pairs.append((genuine_img, pair_img, 1))  # 1 for genuine pair
        
        # Create forged pairs (genuine + forged)
        for genuine_img in self.genuine_images:
            if self.forged_images:
                for _ in range(min(self.pair_per_sample, len(self.forged_images))):
                    forged_img = random.choice(self.forged_images)
                    pairs.append((genuine_img, forged_img, 0))  # 0 for forged pair
        
        # Shuffle pairs
        random.shuffle(pairs)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class TripletSignatureDataset(Dataset):
    """Dataset for triplet loss training"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        
        # Load images grouped by person/signer
        self.signers = self._load_by_signer()
        
        # Create triplets (anchor, positive, negative)
        self.triplets = self._create_triplets()
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_by_signer(self):
        """Group images by signer/person"""
        signers = {}
        
        # Assuming structure: root_dir/genuine/signerX/*.png
        genuine_dir = self.root_dir / 'genuine'
        if genuine_dir.exists():
            for signer_folder in genuine_dir.iterdir():
                if signer_folder.is_dir():
                    signer_name = signer_folder.name
                    signers[signer_name] = {
                        'genuine': list(signer_folder.glob('*.png')) + list(signer_folder.glob('*.jpg')),
                        'forged': []
                    }
        
        # Add forged signatures
        forged_dir = self.root_dir / 'forged'
        if forged_dir.exists():
            for signer_folder in forged_dir.iterdir():
                if signer_folder.is_dir():
                    signer_name = signer_folder.name
                    if signer_name in signers:
                        signers[signer_name]['forged'] = list(signer_folder.glob('*.png')) + list(signer_folder.glob('*.jpg'))
        
        return signers
    
    def _create_triplets(self):
        """Create anchor-positive-negative triplets"""
        triplets = []
        
        for signer, images in self.signers.items():
            genuine_imgs = images['genuine']
            forged_imgs = images['forged']
            
            if len(genuine_imgs) >= 2 and forged_imgs:
                # Create triplets: anchor (genuine), positive (different genuine), negative (forged)
                for anchor_img in genuine_imgs:
                    # Positive: different genuine from same signer
                    positive_imgs = [img for img in genuine_imgs if img != anchor_img]
                    if positive_imgs:
                        positive_img = random.choice(positive_imgs)
                        # Negative: forged signature
                        negative_img = random.choice(forged_imgs)
                        triplets.append((anchor_img, positive_img, negative_img))
        
        random.shuffle(triplets)
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative

def test_dataset():
    """Test the dataset loader"""
    print("Testing SiameseSignatureDataset...")
    
    # Create dummy dataset structure if none exists
    test_dir = Path("data/test_siamese")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "genuine").mkdir(exist_ok=True)
    (test_dir / "forged").mkdir(exist_ok=True)
    
    # Create dummy images
    for i in range(5):
        img = Image.new('RGB', (224, 224), color='white')
        img.save(test_dir / "genuine" / f"genuine_{i}.png")
        
        img = Image.new('RGB', (224, 224), color='white')
        img.save(test_dir / "forged" / f"forged_{i}.png")
    
    # Test dataset
    dataset = SiameseSignatureDataset(test_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    img1, img2, label = dataset[0]
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Label: {label}")
    
    return dataset

if __name__ == "__main__":
    dataset = test_dataset()