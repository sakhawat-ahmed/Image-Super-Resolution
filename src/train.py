import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, 
                epochs=50, model_name='Model', scale_factor=2):
    """Train a single model with validation"""
    
    print(f"\nTraining {model_name} for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    
    # Training history
    history = {
        'train_loss': [],
        'valid_loss': [],
        'epoch_time': [],
        'learning_rate': []
    }
    
    # Learning rate scheduler - FIXED: removed 'verbose' parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create directory for checkpoints
    checkpoint_dir = f"../models/{model_name}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    model.train()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            # Move data to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
            
            for lr_imgs, hr_imgs in valid_pbar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                
                valid_loss += loss.item()
                
                valid_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}'
                })
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['epoch_time'].append(epoch_time)
        history['learning_rate'].append(current_lr)
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Print learning rate reduction if it changed
        if epoch > 0 and history['learning_rate'][-1] != history['learning_rate'][-2]:
            print(f"Learning rate reduced to {current_lr:.6f}")
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, f"{checkpoint_dir}/best_model.pth")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Valid Loss: {valid_loss:.6f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = f"../models/{model_name}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_epoch': best_epoch,
        'best_valid_loss': best_valid_loss
    }, final_path)
    
    print(f"\nBest model at epoch {best_epoch} with validation loss: {best_valid_loss:.6f}")
    print(f"Final model saved: {final_path}")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    return history

def plot_training_history(history, model_name):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['valid_loss'], label='Valid', linewidth=2)
    axes[0, 0].set_title(f'{model_name} - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time plot
    axes[0, 1].plot(history['epoch_time'])
    axes[0, 1].set_title(f'{model_name} - Epoch Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_title(f'{model_name} - Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Empty subplot for future use
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'../results/training_plots/{model_name}_training_history.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save history to file
    history_file = f'../results/training_plots/{model_name}_history.npy'
    np.save(history_file, history)
    print(f"Training history saved to {history_file}")