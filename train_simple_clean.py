import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from data import MatDataset
from models import PhyCRNet  # Simple model
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def simple_mse_loss(pred, target):
    """Simple MSE loss function"""
    return nn.MSELoss()(pred, target)

def train_clean_model():
    """Clean and stable PhyCRNet training"""
    
    print("Clean PhyCRNet Training")
    print("=" * 50)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'num_epochs': 200,  # Shorter for quick testing
        'batch_size': 8,
        'learning_rate': 1e-4,
        'save_interval': 25,
        'data_file': 'Ra_10^5_Rd_1.8.mat',
        'model_save_path': 'clean_model_checkpoint.pth'
    }
    
    print(f"Device: {device}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("=" * 50)
    
    # Create directories
    os.makedirs('clean_training_results', exist_ok=True)
    
    # Dataset
    print("Loading dataset...")
    try:
        dataset = MatDataset(config['data_file'])
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Model
    print("Initializing model...")
    try:
        model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized: {total_params:,} parameters")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{config["num_epochs"]}')
        
        for batch_idx, (input_state, target_state, _) in enumerate(train_progress):
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(input_state)
            loss = loss_fn(prediction, target_state)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            train_progress.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for input_state, target_state, _ in val_loader:
                input_state = input_state.to(device)
                target_state = target_state.to(device)
                
                prediction = model(input_state)
                loss = loss_fn(prediction, target_state)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} ({epoch_time:.1f}s) - "
              f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, config['model_save_path'])
            print(f"  New best model saved! Val loss: {best_val_loss:.6f}")
        
        # Save visualization every save_interval epochs
        if (epoch + 1) % config['save_interval'] == 0:
            save_training_visualization(
                model, val_loader, train_losses, val_losses, epoch+1, device
            )
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved as: {config['model_save_path']}")
    
    return config['model_save_path']

def save_training_visualization(model, val_loader, train_losses, val_losses, epoch, device):
    """Save training visualization"""
    
    # Sample prediction
    model.eval()
    with torch.no_grad():
        for input_state, target_state, _ in val_loader:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            prediction = model(input_state)
            
            # Take first sample from batch
            input_img = input_state[0].cpu().numpy()
            target_img = target_state[0].cpu().numpy()
            pred_img = prediction[0].cpu().numpy()
            
            break
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
    
    field_names = ['U-velocity', 'V-velocity', 'Temperature', 'Pressure']
    cmaps = ['RdBu_r', 'RdBu_r', 'hot', 'viridis']
    
    for i, (field_name, cmap) in enumerate(zip(field_names, cmaps)):
        # Input
        im1 = axes[0, i].imshow(input_img[i], cmap=cmap, aspect='equal', origin='lower')
        axes[0, i].set_title(f'{field_name} (Input)')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Target
        im2 = axes[1, i].imshow(target_img[i], cmap=cmap, aspect='equal', origin='lower')
        axes[1, i].set_title(f'{field_name} (Target)')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
        
        # Prediction
        im3 = axes[2, i].imshow(pred_img[i], cmap=cmap, aspect='equal', origin='lower')
        axes[2, i].set_title(f'{field_name} (Prediction)')
        plt.colorbar(im3, ax=axes[2, i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'clean_training_results/epoch_{epoch:03d}_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'clean_training_results/epoch_{epoch:03d}_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved for epoch {epoch}")

if __name__ == "__main__":
    model_path = train_clean_model() 