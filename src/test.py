import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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

def test_model(model, test_loader, device, scale_factor=2):
    """Test model on test dataset"""
    
    model.eval()
    
    psnr_scores = []
    ssim_scores = []
    image_results = []
    
    with torch.no_grad():
        for idx, (lr_imgs, hr_imgs) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate super-resolved image
            sr_imgs = model(lr_imgs)
            
            # Convert to numpy
            sr_np = sr_imgs.cpu().numpy().squeeze()
            hr_np = hr_imgs.cpu().numpy().squeeze()
            lr_np = lr_imgs.cpu().numpy().squeeze()
            
            # Handle batch dimension
            if len(sr_np.shape) == 4:  # Batch dimension present
                sr_np = sr_np[0]
                hr_np = hr_np[0]
                lr_np = lr_np[0]
            
            # Convert from (C, H, W) to (H, W, C)
            sr_np = np.transpose(sr_np, (1, 2, 0))
            hr_np = np.transpose(hr_np, (1, 2, 0))
            lr_np = np.transpose(lr_np, (1, 2, 0))
            
            # Denormalize to [0, 255]
            sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
            hr_np = np.clip(hr_np * 255, 0, 255).astype(np.uint8)
            lr_np = np.clip(lr_np * 255, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV saving
            sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
            hr_bgr = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)
            lr_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
            
            # Calculate metrics
            current_psnr = calculate_psnr(sr_np, hr_np)
            current_ssim = calculate_ssim(sr_np, hr_np)
            
            psnr_scores.append(current_psnr)
            ssim_scores.append(current_ssim)
            
            # Store sample images for visualization
            if idx < 5:  # Store first 5 samples
                image_results.append((lr_bgr, sr_bgr, hr_bgr))
            
            # Print sample results for first 3 images
            if idx < 3:
                print(f"Sample {idx+1}: PSNR={current_psnr:.2f} dB, SSIM={current_ssim:.4f}")
                
                # Save individual comparison
                save_individual_comparison(lr_bgr, sr_bgr, hr_bgr, idx+1, 
                                         current_psnr, current_ssim, model.__class__.__name__)
    
    return psnr_scores, ssim_scores, image_results

def save_individual_comparison(lr_img, sr_img, hr_img, idx, psnr, ssim, model_name):
    """Save individual image comparison"""
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{model_name} Output\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('High Resolution')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(f'../results/test_results/{model_name}', exist_ok=True)
    
    # Save figure
    plt.savefig(f'../results/test_results/{model_name}/sample_{idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    cv2.imwrite(f'../results/test_results/{model_name}/sample_{idx}_lr.png', lr_img)
    cv2.imwrite(f'../results/test_results/{model_name}/sample_{idx}_sr.png', sr_img)
    cv2.imwrite(f'../results/test_results/{model_name}/sample_{idx}_hr.png', hr_img)