import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import os

class DIV2KDataset(Dataset):
    """DIV2K Dataset for Super-Resolution with fixed dimensions"""
    
    def __init__(self, hr_dir, lr_dir, scale_factor=2, patch_size=48, train=True):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.train = train
        
        # Collect image paths
        self.hr_images = sorted(list(self.hr_dir.glob("*.png")))
        
        if not self.hr_images:
            raise FileNotFoundError(f"No HR images found in {hr_dir}")
        
        # Verify corresponding LR images exist
        self.lr_images = []
        self.valid_indices = []
        
        for idx, hr_path in enumerate(self.hr_images):
            # Try multiple naming conventions
            possible_names = [
                hr_path.name,
                hr_path.name.replace(".png", f"x{scale_factor}.png"),
                hr_path.name.replace(".png", f"x{scale_factor}.png").lower(),
                f"{hr_path.stem}x{scale_factor}.png"
            ]
            
            lr_found = False
            for lr_filename in possible_names:
                lr_path = self.lr_dir / lr_filename
                if lr_path.exists():
                    self.lr_images.append(lr_path)
                    self.valid_indices.append(idx)
                    lr_found = True
                    break
            
            if not lr_found:
                print(f"Warning: LR image not found for {hr_path.name}")
        
        print(f"Loaded {len(self.lr_images)} matching HR-LR image pairs")
    
    def __len__(self):
        return len(self.lr_images)
    
    def load_and_resize(self, hr_path, lr_path, target_size=(256, 256)):
        """Load and resize images to consistent dimensions"""
        # Load HR image
        hr_img = cv2.imread(str(hr_path))
        if hr_img is None:
            raise ValueError(f"Failed to load HR image: {hr_path}")
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Load LR image
        lr_img = cv2.imread(str(lr_path))
        if lr_img is None:
            raise ValueError(f"Failed to load LR image: {lr_path}")
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        hr_img = cv2.resize(hr_img, target_size, interpolation=cv2.INTER_CUBIC)
        lr_img = cv2.resize(lr_img, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        hr_img = hr_img.astype(np.float32) / 255.0
        lr_img = lr_img.astype(np.float32) / 255.0
        
        return lr_img, hr_img
    
    def extract_random_patch(self, lr_img, hr_img):
        """Extract random patch from images for training"""
        h, w = lr_img.shape[:2]
        
        # Ensure patch size is valid
        patch_size = min(self.patch_size, h, w)
        
        # Random starting point
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        
        # Extract patches
        lr_patch = lr_img[y:y+patch_size, x:x+patch_size]
        hr_patch = hr_img[y:y+patch_size, x:x+patch_size]
        
        return lr_patch, hr_patch
    
    def __getitem__(self, idx):
        """Get a single data sample"""
        hr_path = self.hr_images[self.valid_indices[idx]]
        lr_path = self.lr_images[idx]
        
        # Load and resize images to consistent size
        lr_img, hr_img = self.load_and_resize(hr_path, lr_path, target_size=(256, 256))
        
        # Extract random patches for training
        if self.train and self.patch_size is not None:
            lr_img, hr_img = self.extract_random_patch(lr_img, hr_img)
        
        # Convert to PyTorch tensors (C, H, W format)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float().contiguous()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float().contiguous()
        
        return lr_tensor, hr_tensor
    
    def verify_dataset(self):
        """Verify the dataset structure and sample images"""
        print("\nDataset Verification:")
        print(f"HR Directory: {self.hr_dir}")
        print(f"LR Directory: {self.lr_dir}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"Total Images: {len(self)}")
        
        # Check first few images
        for i in range(min(3, len(self))):
            lr, hr = self[i]
            print(f"\nSample {i}:")
            print(f"  LR shape: {lr.shape}")
            print(f"  HR shape: {hr.shape}")
            print(f"  LR range: [{lr.min():.3f}, {lr.max():.3f}]")
            print(f"  HR range: [{hr.min():.3f}, {hr.max():.3f}]")
        
        return True