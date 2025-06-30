"""
Complete Physics-Enhanced PhyCRNet Training
Uses the exact PDE system for natural convection with all physics terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from data import MatDataset
from models import PhyCRNet
from accurate_physics_loss import AccuratePhysicsLoss

def train_complete_physics_model():
    """Train PhyCRNet with the complete PDE system implementation."""
    
    print("Complete Physics-Enhanced PhyCRNet Training")
    print("Implementing Full PDE System:")
    print("   1. Continuity: ∂U/∂X + ∂V/∂Y = 0")
    print("   2. X-momentum: ∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + Pr[∇²U] - (Pr/Da)U")
    print("   3. Y-momentum: ∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + Pr[∇²V] + Ra·Pr·θ - Ha²·Pr·V - (Pr/Da)V")
    print("   4. Energy: ∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (1 + 4Rd/3)[∇²θ] + Q·θ")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'num_epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-5,  # Very conservative
        'save_interval': 20,
        'data_file': 'Ra_10^5_Rd_1.8.mat',
        'model_save_path': 'complete_physics_model_checkpoint.pth',
        'physics_weight_initial': 0.001,  # Start very small
        'physics_weight_max': 0.1,       # Maximum contribution
        'data_weight': 1.0,
        'warmup_epochs': 20,
    }
    
    print(f"Device: {device}")
    print(f"Physics weight: {config['physics_weight_initial']} → {config['physics_weight_max']}")
    
    # Create directories
    os.makedirs('complete_physics_results', exist_ok=True)
    
    # Dataset
    dataset = MatDataset(config['data_file'])
    dataset_params = dataset.get_params()
    
    # Physics parameters are already loaded from MATLAB file
    # No need to override with defaults - use actual values
    
    print(f"Physical Parameters:")
    print(f"   Ra = {dataset_params['Ra']:.1e} (Rayleigh)")
    print(f"   Pr = {dataset_params['Pr']:.3f} (Prandtl)")
    print(f"   Ha = {dataset_params['Ha']:.1f} (Hartmann)")
    print(f"   Da = {dataset_params['Da']:.1e} (Darcy)")
    print(f"   Rd = {dataset_params['Rd']:.2f} (Radiation)")
    print(f"   Q = {dataset_params['Q']:.3f} (Heat source)")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    # Physics loss
    physics_loss = AccuratePhysicsLoss(
        dataset_params, 
        dt=dataset_params.get('dt', 0.0001),
        dx=1.0, dy=1.0
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.8)
    
    # Training tracking
    history = {
        'train_losses': [], 'val_losses': [], 'physics_losses': [], 'data_losses': [],
        'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': []
    }
    best_val_loss = float('inf')
    
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        physics_loss.set_epoch(epoch)
        
        # Progressive physics weight
        if epoch < config['warmup_epochs']:
            progress = epoch / config['warmup_epochs']
            physics_weight = config['physics_weight_initial'] * progress
        else:
            progress = min((epoch - config['warmup_epochs']) / 30.0, 1.0)
            physics_weight = (config['physics_weight_initial'] + 
                            (config['physics_weight_max'] - config['physics_weight_initial']) * progress)
        
        # Training
        model.train()
        epoch_stats = {
            'train_loss': 0.0, 'physics_loss': 0.0, 'data_loss': 0.0,
            'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0, 'energy': 0.0,
            'batches': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
        for input_state, target_state, _ in progress_bar:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(input_state)
            
            # Data loss
            loss_data = F.mse_loss(prediction, target_state)
            
            # Physics loss with components
            try:
                physics_result = physics_loss(input_state, prediction, validation_mode=True)
                loss_physics_total = physics_result['total']
                
                # Combined loss
                total_loss = config['data_weight'] * loss_data + physics_weight * loss_physics_total
                
                # Backward pass
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    
                    # Update statistics
                    epoch_stats['train_loss'] += total_loss.item()
                    epoch_stats['physics_loss'] += loss_physics_total.item()
                    epoch_stats['data_loss'] += loss_data.item()
                    epoch_stats['continuity'] += physics_result['continuity'].item()
                    epoch_stats['momentum_x'] += physics_result['momentum_x'].item()
                    epoch_stats['momentum_y'] += physics_result['momentum_y'].item()
                    epoch_stats['energy'] += physics_result['energy'].item()
                    epoch_stats['batches'] += 1
                
            except Exception as e:
                print(f"Warning - Physics error: {e}")
                # Fallback to data loss only
                loss_data.backward()
                optimizer.step()
                
                epoch_stats['train_loss'] += loss_data.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['batches'] += 1
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}' if 'total_loss' in locals() else f'{loss_data.item():.4f}',
                'Physics': f'{physics_weight:.4f}'
            })
        
        # Average training statistics
        if epoch_stats['batches'] > 0:
            for key in epoch_stats:
                if key != 'batches':
                    epoch_stats[key] /= epoch_stats['batches']
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for input_state, target_state, _ in val_loader:
                input_state = input_state.to(device)
                target_state = target_state.to(device)
                prediction = model(input_state)
                loss = F.mse_loss(prediction, target_state)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_losses'].append(epoch_stats['train_loss'])
        history['val_losses'].append(avg_val_loss)
        history['physics_losses'].append(epoch_stats['physics_loss'])
        history['data_losses'].append(epoch_stats['data_loss'])
        history['continuity'].append(epoch_stats['continuity'])
        history['momentum_x'].append(epoch_stats['momentum_x'])
        history['momentum_y'].append(epoch_stats['momentum_y'])
        history['energy'].append(epoch_stats['energy'])
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1:3d}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: {epoch_stats['train_loss']:.6f} (Data: {epoch_stats['data_loss']:.6f}, Physics: {epoch_stats['physics_loss']:.6f})")
        print(f"  Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}, Physics Weight: {physics_weight:.6f}")
        
        if epoch % 10 == 0:
            print(f"  Physics Components:")
            print(f"     Continuity: {epoch_stats['continuity']:.8f}")
            print(f"     Momentum X: {epoch_stats['momentum_x']:.8f}")
            print(f"     Momentum Y: {epoch_stats['momentum_y']:.8f}")
            print(f"     Energy: {epoch_stats['energy']:.8f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'config': config,
                'dataset_params': dataset_params
            }, config['model_save_path'])
            print(f"  Best model saved! Val loss: {best_val_loss:.6f}")
        
        # Save visualization
        if (epoch + 1) % config['save_interval'] == 0:
            save_complete_visualization(model, val_loader, history, epoch+1, device)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return config['model_save_path']

def save_complete_visualization(model, val_loader, history, epoch, device):
    """Save comprehensive visualization."""
    
    # Get sample prediction
    model.eval()
    with torch.no_grad():
        for input_state, target_state, _ in val_loader:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            prediction = model(input_state)
            
            input_img = input_state[0].cpu().numpy()
            target_img = target_state[0].cpu().numpy()
            pred_img = prediction[0].cpu().numpy()
            break
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Fields (top section)
    field_names = ['U-velocity', 'V-velocity', 'Temperature', 'Pressure']
    cmaps = ['RdBu_r', 'RdBu_r', 'hot', 'viridis']
    
    for i, (field_name, cmap) in enumerate(zip(field_names, cmaps)):
        # Input
        ax = plt.subplot(4, 4, i+1)
        im = ax.imshow(input_img[i], cmap=cmap, aspect='equal', origin='lower')
        ax.set_title(f'{field_name} (Input)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Target
        ax = plt.subplot(4, 4, i+5)
        im = ax.imshow(target_img[i], cmap=cmap, aspect='equal', origin='lower')
        ax.set_title(f'{field_name} (Target)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Prediction
        ax = plt.subplot(4, 4, i+9)
        im = ax.imshow(pred_img[i], cmap=cmap, aspect='equal', origin='lower')
        ax.set_title(f'{field_name} (Prediction)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Error
        ax = plt.subplot(4, 4, i+13)
        error = pred_img[i] - target_img[i]
        im = ax.imshow(error, cmap='RdBu_r', aspect='equal', origin='lower')
        ax.set_title(f'{field_name} (Error)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle(f'Complete Physics Training - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'complete_physics_results/epoch_{epoch:03d}_fields.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loss analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # type: ignore
    
    # Total losses
    axes[0, 0].plot(history['train_losses'], label='Train', linewidth=2)  # type: ignore
    axes[0, 0].plot(history['val_losses'], label='Validation', linewidth=2)  # type: ignore
    axes[0, 0].set_title('Total Loss')  # type: ignore
    axes[0, 0].legend()  # type: ignore
    axes[0, 0].grid(True, alpha=0.3)  # type: ignore
    
    # Component losses
    axes[0, 1].plot(history['data_losses'], label='Data', linewidth=2)  # type: ignore
    axes[0, 1].plot(history['physics_losses'], label='Physics', linewidth=2)  # type: ignore
    axes[0, 1].set_title('Component Losses')  # type: ignore
    axes[0, 1].legend()  # type: ignore
    axes[0, 1].grid(True, alpha=0.3)  # type: ignore
    
    # Physics terms
    axes[0, 2].plot(history['continuity'], label='Continuity', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['momentum_x'], label='Momentum X', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['momentum_y'], label='Momentum Y', linewidth=2)  # type: ignore
    axes[0, 2].plot(history['energy'], label='Energy', linewidth=2)  # type: ignore
    axes[0, 2].set_title('Physics Terms')  # type: ignore
    axes[0, 2].legend()  # type: ignore
    axes[0, 2].grid(True, alpha=0.3)  # type: ignore
    
    # Log scale versions
    axes[1, 0].semilogy(history['train_losses'], label='Train', linewidth=2)  # type: ignore
    axes[1, 0].semilogy(history['val_losses'], label='Validation', linewidth=2)  # type: ignore
    axes[1, 0].set_title('Total Loss (Log)')  # type: ignore
    axes[1, 0].legend()  # type: ignore
    axes[1, 0].grid(True, alpha=0.3)  # type: ignore
    
    axes[1, 1].semilogy(history['data_losses'], label='Data', linewidth=2)  # type: ignore
    axes[1, 1].semilogy([max(x, 1e-10) for x in history['physics_losses']], label='Physics', linewidth=2)  # type: ignore
    axes[1, 1].set_title('Component Losses (Log)')  # type: ignore
    axes[1, 1].legend()  # type: ignore
    axes[1, 1].grid(True, alpha=0.3)  # type: ignore
    
    # Recent physics terms
    recent = max(1, len(history['continuity']) - 20)
    axes[1, 2].plot(range(recent, len(history['continuity'])), history['continuity'][recent:],   # type: ignore
                   label='Continuity', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['momentum_x'])), history['momentum_x'][recent:],   # type: ignore
                   label='Momentum X', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['momentum_y'])), history['momentum_y'][recent:],   # type: ignore
                   label='Momentum Y', linewidth=2)
    axes[1, 2].plot(range(recent, len(history['energy'])), history['energy'][recent:],   # type: ignore
                   label='Energy', linewidth=2)
    axes[1, 2].set_title('Recent Physics Terms')  # type: ignore
    axes[1, 2].legend()  # type: ignore
    axes[1, 2].grid(True, alpha=0.3)  # type: ignore
    
    plt.suptitle(f'Complete Physics Analysis - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'complete_physics_results/epoch_{epoch:03d}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_complete_physics_model() 