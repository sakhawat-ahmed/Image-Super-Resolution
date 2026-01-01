import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same dimensions. Got {img1.shape} and {img2.shape}")
    
    # Calculate MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return 100
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Use skimage's SSIM for better accuracy
    try:
        # Ensure images are in correct format
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            # Convert to grayscale for SSIM calculation (common practice)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            return ssim_skimage(img1_gray, img2_gray, data_range=255)
        else:
            return ssim_skimage(img1, img2, data_range=255)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0

def plot_results(lr_images, sr_images, hr_images, titles=None, save_path=None):
    """Plot comparison results"""
    if titles is None:
        titles = ['Low Resolution', 'Super Resolution', 'High Resolution']
    
    num_samples = min(3, len(lr_images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # LR Image
        axes[i, 0].imshow(cv2.cvtColor(lr_images[i], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'{titles[0]} - Sample {i+1}')
        axes[i, 0].axis('off')
        
        # SR Image
        axes[i, 1].imshow(cv2.cvtColor(sr_images[i], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'{titles[1]} - Sample {i+1}')
        axes[i, 1].axis('off')
        
        # HR Image
        axes[i, 2].imshow(cv2.cvtColor(hr_images[i], cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f'{titles[2]} - Sample {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def save_model_summary(model, filepath="model_summary.txt"):
    """Save model architecture summary to file"""
    with open(filepath, 'w') as f:
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write("="*50 + "\n\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n\n")
        
        # Layer details
        f.write("Layer Details:\n")
        f.write("-"*50 + "\n")
        
        for name, module in model.named_modules():
            if name:  # Skip empty name (the whole model)
                num_params = sum(p.numel() for p in module.parameters())
                f.write(f"{name}: {module.__class__.__name__}, Parameters: {num_params:,}\n")
    
    print(f"Model summary saved to {filepath}")

def check_dataset_structure():
    """Check if dataset structure is correct"""
    base_path = Path("../DIV2KDataset")
    
    required_folders = [
        "DIV2K_train_HR",
        "DIV2K_train_LR_bicubic",
        "DIV2K_valid_HR",
        "DIV2K_valid_LR_bicubic"
    ]
    
    print("Checking dataset structure...")
    
    missing_folders = []
    for folder in required_folders:
        folder_path = base_path / folder
        if not folder_path.exists():
            missing_folders.append(folder)
            print(f"❌ Missing: {folder}")
        else:
            # Count images
            png_files = list(folder_path.rglob("*.png"))
            print(f"✓ {folder}: {len(png_files)} images")
            
            # Check for X2, X3, X4 subfolders for LR
            if "LR_bicubic" in folder:
                for scale in ["X2", "X3", "X4"]:
                    scale_path = folder_path / scale
                    if scale_path.exists():
                        scale_images = list(scale_path.glob("*.png"))
                        print(f"  ├── {scale}: {len(scale_images)} images")
    
    if missing_folders:
        print(f"\n❌ Missing folders: {missing_folders}")
        return False
    else:
        print("\n✅ Dataset structure is correct!")
        return True

def prepare_sample_images(num_samples=5):
    """Prepare sample images for demonstration"""
    base_path = Path("../DIV2KDataset")
    
    # Get sample images from validation set
    hr_dir = base_path / "DIV2K_valid_HR"
    lr_dir = base_path / "DIV2K_valid_LR_bicubic" / "X2"
    
    hr_images = sorted(list(hr_dir.glob("*.png")))[:num_samples]
    sample_images = []
    
    for hr_path in hr_images:
        lr_filename = hr_path.name.replace(".png", "x2.png")
        lr_path = lr_dir / lr_filename
        
        if lr_path.exists():
            # Load images
            hr_img = cv2.imread(str(hr_path))
            lr_img = cv2.imread(str(lr_path))
            
            sample_images.append({
                'lr': lr_img,
                'hr': hr_img,
                'name': hr_path.stem
            })
    
    print(f"Prepared {len(sample_images)} sample images")
    return sample_images