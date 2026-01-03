import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import sys
import os
import traceback
import json

# Import custom modules
from data_preprocessing import DIV2KDataset
from srcnn_model import SRCNN
from edsr_model import EDSR

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Ensure images are the same size
    if img1.shape != img2.shape:
        # Resize to smaller dimension
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
    
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    if mse == 0:
        return 100
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    try:
        # Ensure images are the same size
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            from skimage.metrics import structural_similarity as ssim_skimage
            return ssim_skimage(img1_gray, img2_gray, data_range=255)
        else:
            from skimage.metrics import structural_similarity as ssim_skimage
            return ssim_skimage(img1, img2, data_range=255)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0.0

class SuperResolutionSystem:
    """Complete system for image super-resolution"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        self.histories = {}
        
        # Create necessary directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for the project"""
        directories = ['../models', '../results', '../results/training_plots', 
                      '../results/test_results', '../results/sample_images']
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self):
        """Prepare training and testing datasets"""
        print("Preparing datasets...")
        
        try:
            # Create datasets
            train_dataset = DIV2KDataset(
                hr_dir=self.config.train_hr_path,
                lr_dir=self.config.train_lr_path,
                scale_factor=self.config.scale_factor,
                patch_size=self.config.patch_size,
                train=True
            )
            
            valid_dataset = DIV2KDataset(
                hr_dir=self.config.valid_hr_path,
                lr_dir=self.config.valid_lr_path,
                scale_factor=self.config.scale_factor,
                patch_size=None,  # No patching for validation
                train=False
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(valid_dataset)}")
            
            return train_loader, valid_loader
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise
    
    def build_models(self):
        """Initialize models"""
        print("Building models...")
        
        # SRCNN Model
        self.models['SRCNN'] = SRCNN(
            num_channels=3,
            base_filter=64
        ).to(self.device)
        
        # EDSR Model (fixed for same input/output size)
        self.models['EDSR'] = EDSR(
            num_channels=3,
            num_features=32,
            num_blocks=4,
            res_scale=0.1
        ).to(self.device)
        
        print(f"\nModel Architectures:")
        print(f"{'='*40}")
        for name, model in self.models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"{name}: {params:,} parameters")
    
    def train_model(self, model, train_loader, valid_loader, criterion, optimizer, epochs, name):
        """Train a single model"""
        print(f"\nTraining {name} for {epochs} epochs...")
        
        history = {
            'train_loss': [],
            'valid_loss': [],
            'train_psnr': [],
            'valid_psnr': []
        }
        
        best_valid_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_psnr = 0.0
            
            for lr_imgs, hr_imgs in train_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate PSNR for this batch
                with torch.no_grad():
                    mse = torch.mean((sr_imgs - hr_imgs) ** 2)
                    batch_psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else 50
                    train_psnr += batch_psnr.item()
            
            # Validation
            model.eval()
            valid_loss = 0.0
            valid_psnr = 0.0
            
            with torch.no_grad():
                for lr_imgs, hr_imgs in valid_loader:
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
                    sr_imgs = model(lr_imgs)
                    loss = criterion(sr_imgs, hr_imgs)
                    valid_loss += loss.item()
                    
                    # Calculate PSNR
                    mse = torch.mean((sr_imgs - hr_imgs) ** 2)
                    batch_psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else 50
                    valid_psnr += batch_psnr.item()
            
            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            train_psnr /= len(train_loader)
            valid_psnr /= len(valid_loader)
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_psnr'].append(train_psnr)
            history['valid_psnr'].append(valid_psnr)
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = model.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, "
                  f"Valid PSNR: {valid_psnr:.2f} dB")
        
        # Save best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model
        model_path = f"../models/{name}_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': self.config.__dict__
        }, model_path)
        print(f"Model saved: {model_path}")
        
        return history
    
    def evaluate_model(self, model, valid_loader, name):
        """Evaluate a single model"""
        print(f"\nEvaluating {name}...")
        
        model.eval()
        psnr_scores = []
        ssim_scores = []
        sample_images = []
        
        with torch.no_grad():
            for idx, (lr_imgs, hr_imgs) in enumerate(valid_loader):
                if idx >= 20:  # Limit to 20 samples for speed
                    break
                
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                sr_imgs = model(lr_imgs)
                
                # Convert to numpy
                sr_np = sr_imgs.cpu().numpy().squeeze()
                hr_np = hr_imgs.cpu().numpy().squeeze()
                lr_np = lr_imgs.cpu().numpy().squeeze()
                
                # Handle dimensions
                if len(sr_np.shape) == 4:
                    sr_np = sr_np[0]
                    hr_np = hr_np[0]
                    lr_np = lr_np[0]
                
                # Convert from (C, H, W) to (H, W, C)
                sr_np = np.transpose(sr_np, (1, 2, 0))
                hr_np = np.transpose(hr_np, (1, 2, 0))
                lr_np = np.transpose(lr_np, (1, 2, 0))
                
                # Denormalize
                sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
                hr_np = np.clip(hr_np * 255, 0, 255).astype(np.uint8)
                lr_np = np.clip(lr_np * 255, 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
                hr_bgr = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)
                lr_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_np, hr_np)
                ssim = calculate_ssim(sr_np, hr_np)
                
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                
                if idx < 5:  # Save first 5 samples
                    sample_images.append((lr_bgr, sr_bgr, hr_bgr))
                
                if idx < 3:
                    print(f"  Sample {idx+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        
        if psnr_scores:
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            std_psnr = np.std(psnr_scores)
            std_ssim = np.std(ssim_scores)
            
            print(f"\n{name} Results:")
            print(f"  Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"  Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
            print(f"  Best PSNR: {max(psnr_scores):.2f} dB")
            print(f"  Best SSIM: {max(ssim_scores):.4f}")
            print(f"  Worst PSNR: {min(psnr_scores):.2f} dB")
            print(f"  Worst SSIM: {min(ssim_scores):.4f}")
            
            return {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'psnr_scores': psnr_scores,
                'ssim_scores': ssim_scores,
                'std_psnr': std_psnr,
                'std_ssim': std_ssim,
                'images': sample_images
            }
        
        return None
    
    def save_sample_results(self, results):
        """Save sample output images and visualizations"""
        print("\nGenerating visualizations...")
        
        for model_name, metrics in results.items():
            if metrics and 'images' in metrics and metrics['images']:
                # Save individual samples
                for i, (lr_img, sr_img, hr_img) in enumerate(metrics['images'][:3]):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
                    axes[0].set_title('Low Resolution')
                    axes[0].axis('off')
                    
                    axes[1].imshow(cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB))
                    axes[1].set_title(f'{model_name} Output\nPSNR: {metrics["psnr"]:.2f} dB\nSSIM: {metrics["ssim"]:.4f}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
                    axes[2].set_title('High Resolution')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'../results/sample_images/{model_name}_sample_{i+1}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
        
        # Create comparison plot if we have multiple models
        if len(results) > 1:
            models_with_results = [m for m in results.keys() if results[m] is not None]
            if len(models_with_results) > 1:
                self.create_comparison_plot(results, models_with_results)
    
    def create_comparison_plot(self, results, model_names):
        """Create comprehensive comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # PSNR comparison
        psnr_values = [results[m]['psnr'] for m in model_names]
        psnr_errors = [results[m]['std_psnr'] for m in model_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        x_pos = np.arange(len(model_names))
        
        axes[0, 0].bar(x_pos, psnr_values, yerr=psnr_errors, capsize=5, color=colors, alpha=0.7)
        axes[0, 0].set_title('PSNR Comparison (dB)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('PSNR (dB)', fontsize=12)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(psnr_values):
            axes[0, 0].text(i, v + 0.2, f'{v:.2f}', ha='center', fontsize=10)
        
        # SSIM comparison
        ssim_values = [results[m]['ssim'] for m in model_names]
        ssim_errors = [results[m]['std_ssim'] for m in model_names]
        
        axes[0, 1].bar(x_pos, ssim_values, yerr=ssim_errors, capsize=5, color=colors, alpha=0.7)
        axes[0, 1].set_title('SSIM Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('SSIM', fontsize=12)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names, fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(ssim_values):
            axes[0, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)
        
        # Training loss comparison
        for i, model_name in enumerate(model_names):
            if model_name in self.histories and 'train_loss' in self.histories[model_name]:
                axes[0, 2].plot(self.histories[model_name]['train_loss'], 
                               label=f'{model_name} Train', color=colors[i], linewidth=2)
                axes[0, 2].plot(self.histories[model_name]['valid_loss'], 
                               label=f'{model_name} Valid', color=colors[i], 
                               linestyle='--', linewidth=2)
        
        if axes[0, 2].has_data():
            axes[0, 2].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch', fontsize=12)
            axes[0, 2].set_ylabel('Loss', fontsize=12)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # PSNR distribution
        psnr_data = [results[m]['psnr_scores'] for m in model_names]
        box = axes[1, 0].boxplot(psnr_data, tick_labels=model_names, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # SSIM distribution
        ssim_data = [results[m]['ssim_scores'] for m in model_names]
        box = axes[1, 1].boxplot(ssim_data, tick_labels=model_names, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('SSIM', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Sample image montage - FIXED size mismatch
        if model_names and results[model_names[0]] and results[model_names[0]]['images']:
            sample_idx = 0
            lr_img, _, hr_img = results[model_names[0]]['images'][sample_idx]
            
            # Ensure all images are the same size
            target_size = (256, 256)  # Standard size
            
            # Create montage with all model outputs
            images_to_stack = []
            
            # LR image (resized)
            lr_resized = cv2.resize(lr_img, target_size)
            images_to_stack.append(lr_resized)
            
            # Add all model outputs (resized)
            for model_name in model_names:
                if results[model_name] and results[model_name]['images']:
                    _, sr_img, _ = results[model_name]['images'][sample_idx]
                    sr_resized = cv2.resize(sr_img, target_size)
                    images_to_stack.append(sr_resized)
            
            # HR image (resized)
            hr_resized = cv2.resize(hr_img, target_size)
            images_to_stack.append(hr_resized)
            
            # Create montage
            montage = np.hstack(images_to_stack)
            axes[1, 2].imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
            
            # Create labels
            labels = ['LR'] + model_names + ['HR']
            x_positions = np.linspace(50, target_size[1] * len(labels) - 50, len(labels))
            
            for i, (label, x_pos) in enumerate(zip(labels, x_positions)):
                axes[1, 2].text(x_pos, 30, label, color='white', fontsize=11,
                              fontweight='bold', backgroundcolor='black', ha='center')
            
            axes[1, 2].set_title('Sample Image Comparison', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Comparison plot saved to ../results/model_comparison.png")
    
    def generate_final_report(self, results):
        """Generate comprehensive final report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text report
        report_file = f"../results/final_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" " * 20 + "FINAL PROJECT REPORT\n")
            f.write(" " * 15 + "Image Super-Resolution System\n")
            f.write(" " * 10 + "Principles and Platforms of Deep Learning\n")
            f.write(" " * 20 + "Fall Semester 2025\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. PROJECT OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Scale Factor: {self.config.scale_factor}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total Training Images: 800\n")
            f.write(f"Total Validation Images: 100\n")
            f.write(f"Patch Size: {self.config.patch_size}\n")
            f.write(f"Batch Size: {self.config.batch_size}\n")
            f.write(f"Learning Rate: {self.config.learning_rate}\n\n")
            
            f.write("2. MODEL ARCHITECTURES\n")
            f.write("-"*70 + "\n")
            for name, model in self.models.items():
                params = sum(p.numel() for p in model.parameters())
                f.write(f"{name}:\n")
                f.write(f"  Parameters: {params:,}\n")
                f.write(f"  Epochs Trained: {self.config.srcnn_epochs if 'SRCNN' in name else self.config.edsr_epochs}\n\n")
            
            f.write("3. EXPERIMENTAL RESULTS\n")
            f.write("-"*70 + "\n")
            
            if results:
                for name, metrics in results.items():
                    if metrics:
                        f.write(f"\n{name} Performance:\n")
                        f.write(f"  Average PSNR: {metrics['psnr']:.2f} ± {metrics['std_psnr']:.2f} dB\n")
                        f.write(f"  Average SSIM: {metrics['ssim']:.4f} ± {metrics['std_ssim']:.4f}\n")
                        f.write(f"  Best PSNR: {max(metrics['psnr_scores']):.2f} dB\n")
                        f.write(f"  Best SSIM: {max(metrics['ssim_scores']):.4f}\n")
                        f.write(f"  PSNR Range: {min(metrics['psnr_scores']):.2f} - {max(metrics['psnr_scores']):.2f} dB\n")
                        f.write(f"  SSIM Range: {min(metrics['ssim_scores']):.4f} - {max(metrics['ssim_scores']):.4f}\n")
            
            f.write("\n4. CONCLUSION\n")
            f.write("-"*70 + "\n")
            if results:
                best_model = max(results.keys(), 
                               key=lambda x: results[x]['psnr'] if results[x] else 0)
                f.write(f"The {best_model} model achieved the best performance with ")
                if results[best_model]:
                    f.write(f"{results[best_model]['psnr']:.2f} dB PSNR and ")
                    f.write(f"{results[best_model]['ssim']:.4f} SSIM.\n")
                f.write("Both models demonstrate the effectiveness of deep learning ")
                f.write("approaches for image super-resolution tasks.\n")
            
            f.write("\n5. FILES GENERATED\n")
            f.write("-"*70 + "\n")
            f.write("• Trained models (in ../models/):\n")
            for name in self.models.keys():
                f.write(f"  - {name}_final.pth\n")
            f.write("\n• Results (in ../results/):\n")
            f.write("  - model_comparison.png (visual comparison)\n")
            f.write("  - sample_images/ (sample outputs)\n")
            f.write("  - final_report_*.txt (this report)\n")
            f.write("  - training_plots/ (training history)\n")
        
        print(f"\nReport saved to {report_file}")
        
        # JSON report for programmatic access
        json_report = {
            'timestamp': timestamp,
            'config': self.config.__dict__,
            'device': str(self.device),
            'results': {}
        }
        
        for name, metrics in results.items():
            if metrics:
                json_report['results'][name] = {
                    'psnr': float(metrics['psnr']),
                    'ssim': float(metrics['ssim']),
                    'std_psnr': float(metrics['std_psnr']),
                    'std_ssim': float(metrics['std_ssim'])
                }
        
        json_file = f"../results/results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"JSON results saved to {json_file}")
    
    def run_pipeline(self):  # Changed from run() to run_pipeline()
        """Run the complete pipeline"""
        print("="*70)
        print(" " * 20 + "Image Super-Resolution Project")
        print(" " * 15 + "Principles and Platforms of Deep Learning")
        print(" " * 25 + "Fall Semester 2025")
        print("="*70)
        
        print(f"\nConfiguration:")
        print(f"  Scale Factor: {self.config.scale_factor}")
        print(f"  Device: {self.device}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Epochs: SRCNN={self.config.srcnn_epochs}, EDSR={self.config.edsr_epochs}")
        print(f"  Patch Size: {self.config.patch_size}")
        
        try:
            # Prepare data
            train_loader, valid_loader = self.prepare_data()
            
            # Build models
            self.build_models()
            
            results = {}
            
            # Train and evaluate SRCNN
            try:
                print("\n" + "="*50)
                print("SRCNN TRAINING")
                print("="*50)
                
                srcnn_optimizer = optim.Adam(
                    self.models['SRCNN'].parameters(), 
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999)
                )
                srcnn_criterion = nn.MSELoss()
                
                srcnn_history = self.train_model(
                    self.models['SRCNN'], train_loader, valid_loader,
                    srcnn_criterion, srcnn_optimizer,
                    epochs=self.config.srcnn_epochs, name='SRCNN'
                )
                self.histories['SRCNN'] = srcnn_history
                
                srcnn_results = self.evaluate_model(self.models['SRCNN'], valid_loader, 'SRCNN')
                if srcnn_results:
                    results['SRCNN'] = srcnn_results
                
            except Exception as e:
                print(f"\nError training SRCNN: {e}")
                traceback.print_exc()
            
            # Train and evaluate EDSR
            try:
                print("\n" + "="*50)
                print("EDSR TRAINING")
                print("="*50)
                
                edsr_optimizer = optim.Adam(
                    self.models['EDSR'].parameters(), 
                    lr=self.config.learning_rate * 0.5,
                    betas=(0.9, 0.999)
                )
                edsr_criterion = nn.L1Loss()  # EDSR uses L1 loss
                
                edsr_history = self.train_model(
                    self.models['EDSR'], train_loader, valid_loader,
                    edsr_criterion, edsr_optimizer,
                    epochs=self.config.edsr_epochs, name='EDSR'
                )
                self.histories['EDSR'] = edsr_history
                
                edsr_results = self.evaluate_model(self.models['EDSR'], valid_loader, 'EDSR')
                if edsr_results:
                    results['EDSR'] = edsr_results
                
            except Exception as e:
                print(f"\nError training EDSR: {e}")
                traceback.print_exc()
            
            # Save sample results and generate visualizations
            if results:
                self.save_sample_results(results)
            
            # Generate final report
            self.generate_final_report(results)
            
            print("\n" + "="*70)
            print(" " * 25 + "PROJECT COMPLETED")
            print("="*70)
            
            if results:
                print("\nFINAL RESULTS SUMMARY:")
                print("-" * 70)
                print(f"{'Model':<10} {'PSNR (dB)':<15} {'SSIM':<15} {'Parameters':<15}")
                print("-" * 70)
                
                for name, metrics in results.items():
                    if metrics:
                        params = sum(p.numel() for p in self.models[name].parameters())
                        print(f"{name:<10} {metrics['psnr']:<15.2f} {metrics['ssim']:<15.4f} {params:,}")
                
                print("-" * 70)
                
                # Determine best model
                best_model = max(results.keys(), 
                               key=lambda x: results[x]['psnr'] if results[x] else 0)
                if results[best_model]:
                    print(f"\nBest Model: {best_model}")
                    print(f"  PSNR: {results[best_model]['psnr']:.2f} dB")
                    print(f"  SSIM: {results[best_model]['ssim']:.4f}")
            
            print(f"\nGenerated files saved in ../results/")
            print(f"Models saved in ../models/")
            
        except Exception as e:
            print(f"\nFatal error: {e}")
            traceback.print_exc()

def get_config(scale_factor=2):
    """Get configuration parameters"""
    class Config:
        def __init__(self, scale_factor):
            # Dataset paths
            self.base_dataset_path = "../DIV2KDataset"
            
            # Training paths
            self.train_hr_path = f"{self.base_dataset_path}/DIV2K_train_HR"
            self.train_lr_path = f"{self.base_dataset_path}/DIV2K_train_LR_bicubic/X{scale_factor}"
            
            # Validation paths
            self.valid_hr_path = f"{self.base_dataset_path}/DIV2K_valid_HR"
            self.valid_lr_path = f"{self.base_dataset_path}/DIV2K_valid_LR_bicubic/X{scale_factor}"
            
            # Model parameters
            self.scale_factor = scale_factor
            self.patch_size = 64
            self.batch_size = 8
            
            # Training parameters
            self.learning_rate = 0.001
            self.srcnn_epochs = 10
            self.edsr_epochs = 10
    
    return Config(scale_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Super-Resolution Project')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4],
                       help='Super-resolution scale factor (2, 3, or 4)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs per model')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with reduced epochs')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(scale_factor=args.scale)
    
    # Adjust for quick test mode
    if args.quick:
        config.srcnn_epochs = 3
        config.edsr_epochs = 3
        config.batch_size = 4
        config.patch_size = 32
        print("\n" + "!"*50)
        print("QUICK TEST MODE: Reduced epochs and batch size")
        print("!"*50)
    
    # Override with command line arguments
    if args.epochs != 10:
        config.srcnn_epochs = args.epochs
        config.edsr_epochs = args.epochs
    
    if args.batch != 8:
        config.batch_size = args.batch
    
    # Run system
    system = SuperResolutionSystem(config)
    system.run_pipeline()  # Changed from system.run()