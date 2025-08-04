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
        'num_epochs': 200,
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
    
    # Split dataset by time step (not randomly)
    # Use early time steps for training, later time steps for validation
    train_size = int(0.8 * len(dataset))
    
    # Create indices for time-based split
    train_indices = list(range(train_size))  # Early time steps (0 to train_size-1)
    val_indices = list(range(train_size, len(dataset)))  # Later time steps
    
    # Create subsets based on time indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model with Rd prediction enabled
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1, predict_rd=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters (with Rd prediction)")
    print(f"Target Rd value: {dataset_params['Rd']:.3f}")
    
    # Physics loss configuration
    use_predicted_rd = True  # Set to True to use predicted Rd in physics equations
    enable_physics_loss = True  # Set to True to enable physics loss
    
    physics_loss = AccuratePhysicsLoss(
        dataset_params, 
        dt=dataset_params.get('dt', 0.0001),
        dx=1.0, dy=1.0,
        use_predicted_rd=use_predicted_rd
    ).to(device) if enable_physics_loss else None
    
    print(f"Physics loss enabled: {enable_physics_loss}")
    if not enable_physics_loss:
        print("Running in data-only mode for stability")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.8)
    
    # Training tracking
    history = {
        'train_losses': [], 'val_losses': [], 'physics_losses': [], 'data_losses': [],
        'continuity': [], 'momentum_x': [], 'momentum_y': [], 'energy': [],
        'predicted_rd': [], 'rd_loss': []
    }
    best_val_loss = float('inf')
    worst_val_loss = 0.0
    worst_val_rd = 0.0
    
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        if physics_loss is not None:
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
            'train_loss': 0.0, 'physics_loss': 0.0, 'data_loss': 0.0, 'rd_loss': 0.0,
            'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0, 'energy': 0.0,
            'predicted_rd': 0.0, 'batches': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
        for input_state, target_state, _ in progress_bar:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction, rd_scalar = model(input_state)  # Get both prediction and Rd scalar
            
            # Create extended target with Rd values
            batch_size = target_state.size(0)
            target_rd = torch.full((batch_size, 1), dataset_params['Rd'], 
                                 device=device, dtype=target_state.dtype)
            
            # Extend target_state to include Rd channel
            target_rd_spatial = target_rd.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *target_state.shape[2:])
            target_extended = torch.cat([target_state, target_rd_spatial], dim=1)
            
            # Data loss (for main fields U, V, T, P)
            loss_data_main = F.mse_loss(prediction[:, :4], target_state)
            
            # Rd prediction loss
            loss_rd = F.mse_loss(rd_scalar, target_rd)
            
            # Combined data loss
            loss_data = loss_data_main + 0.1 * loss_rd  # Weight Rd loss lower
            
            # Physics loss with components (if enabled)
            physics_result = None
            loss_physics_total = torch.tensor(0.0, device=device)
            total_loss = config['data_weight'] * loss_data  # Default to data loss only
            
            if enable_physics_loss and physics_loss is not None:
                try:
                    if use_predicted_rd:
                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True, rd_scalar=rd_scalar)
                    else:
                        physics_result = physics_loss(input_state, prediction[:, :4], validation_mode=True)
                        
                    if physics_result is not None and isinstance(physics_result, dict) and 'total' in physics_result:
                        loss_physics_total = physics_result['total']
                        # Ensure physics loss is a scalar tensor
                        if hasattr(loss_physics_total, 'dim') and loss_physics_total.dim() > 0:
                            loss_physics_total = loss_physics_total.mean()
                        # Add physics loss to total
                        total_loss = config['data_weight'] * loss_data + physics_weight * loss_physics_total
                    else:
                        # Physics result is invalid
                        raise ValueError(f"Physics result is invalid: {type(physics_result)}")
                        
                except Exception as e:
                    print(f"Warning - Physics error: {e}")
                    # Fall back to data-only loss
                    total_loss = config['data_weight'] * loss_data
                    physics_result = None
                    loss_physics_total = torch.tensor(0.0, device=device)
            
            # Backward pass
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Update statistics
                epoch_stats['train_loss'] += total_loss.item()
                epoch_stats['physics_loss'] += loss_physics_total.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['rd_loss'] += loss_rd.item()
                
                # Add physics component stats if available
                if physics_result is not None and isinstance(physics_result, dict):
                    try:
                        for key in ['continuity', 'momentum_x', 'momentum_y', 'energy']:
                            if key in physics_result:
                                value = physics_result[key]
                                if hasattr(value, 'dim') and value.dim() == 0:
                                    epoch_stats[key] += value.item()
                                elif hasattr(value, 'mean'):
                                    epoch_stats[key] += value.mean().item()
                                else:
                                    epoch_stats[key] += float(value)
                    except Exception as e:
                        print(f"Warning - Error extracting physics components: {e}")
                
                epoch_stats['predicted_rd'] += rd_scalar.mean().item()
                epoch_stats['batches'] += 1
            else:
                print(f"Warning - Invalid total loss (NaN or Inf): {total_loss}")
                # Use data loss as fallback
                loss_data.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_stats['train_loss'] += loss_data.item()
                epoch_stats['data_loss'] += loss_data.item()
                epoch_stats['rd_loss'] += loss_rd.item()
                epoch_stats['predicted_rd'] += rd_scalar.mean().item()
                epoch_stats['batches'] += 1
            
            # Update progress bar
            current_loss = total_loss.item() if 'total_loss' in locals() and total_loss is not None else loss_data.item()
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Physics': f'{physics_weight:.4f}',
                'Rd': f'{rd_scalar.mean().item():.3f}'
            })
        
        # Average training statistics
        if epoch_stats['batches'] > 0:
            for key in epoch_stats:
                if key != 'batches':
                    epoch_stats[key] /= epoch_stats['batches']
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_rd_avg = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for input_state, target_state, _ in val_loader:
                input_state = input_state.to(device)
                target_state = target_state.to(device)
                prediction, rd_scalar = model(input_state)
                
                # Validation loss (only main fields)
                loss = F.mse_loss(prediction[:, :4], target_state)
                val_loss += loss.item()
                val_rd_avg += rd_scalar.mean().item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        avg_val_rd = val_rd_avg / val_batches if val_batches > 0 else 0.0
        
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
        history['predicted_rd'].append(epoch_stats['predicted_rd'])
        history['rd_loss'].append(epoch_stats['rd_loss'])
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1:3d}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: {epoch_stats['train_loss']:.6f} (Data: {epoch_stats['data_loss']:.6f}, Physics: {epoch_stats['physics_loss']:.6f}, Rd: {epoch_stats['rd_loss']:.6f})")
        print(f"  Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}, Physics Weight: {physics_weight:.6f}")
        print(f"  Predicted Rd: {epoch_stats['predicted_rd']:.4f} (Target: {dataset_params['Rd']:.4f}, Val: {avg_val_rd:.4f})")
        
        if epoch % 10 == 0:
            print(f"  Physics Components:")
            print(f"     Continuity: {epoch_stats['continuity']:.8f}")
            print(f"     Momentum X: {epoch_stats['momentum_x']:.8f}")
            print(f"     Momentum Y: {epoch_stats['momentum_y']:.8f}")
            print(f"     Energy: {epoch_stats['energy']:.8f}")
        
        # Track worst validation loss and corresponding Rd
        if avg_val_loss > worst_val_loss:
            worst_val_loss = avg_val_loss
            worst_val_rd = avg_val_rd
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'worst_val_loss': worst_val_loss,
                'worst_val_rd': worst_val_rd,
                'config': config,
                'dataset_params': dataset_params
            }, config['model_save_path'])
            print(f"  Best model saved! Val loss: {best_val_loss:.6f}")
        
        # Save visualization
        if (epoch + 1) % config['save_interval'] == 0:
            save_complete_visualization(model, val_loader, history, epoch+1, device)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final predicted Rd: {history['predicted_rd'][-1]:.4f} (Target: {dataset_params['Rd']:.4f})")
    return config['model_save_path']

def save_complete_visualization(model, val_loader, history, epoch, device):
    """Save comprehensive visualization."""
    
    # Get sample prediction
    model.eval()
    predicted_rd_values = []
    with torch.no_grad():
        for input_state, target_state, _ in val_loader:
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            prediction, rd_scalar = model(input_state)
            
            input_img = input_state[0].cpu().numpy()
            target_img = target_state[0].cpu().numpy()
            pred_img = prediction[0, :4].cpu().numpy()  # Only main fields for visualization
            predicted_rd_values.append(rd_scalar.cpu().numpy())
            break
    
    # Calculate Rd statistics
    avg_predicted_rd = float(np.mean(predicted_rd_values[0]))
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Fields (top section)
    field_names = ['U-velocity', 'V-velocity', 'Temperature', 'Pressure']
    cmaps = ['seismic', 'seismic', 'seismic', 'seismic']
    
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
        im = ax.imshow(error, cmap='seismic', aspect='equal', origin='lower')
        ax.set_title(f'{field_name} (Error)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle(f'Complete Physics Training - Epoch {epoch}\nPredicted Rd: {avg_predicted_rd:.4f} (Target: 1.8)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'complete_physics_results/epoch_{epoch:03d}_fields.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loss analysis - expand to include Rd tracking
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # type: ignore
    
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
    
    # Rd prediction tracking (new row)
    if 'predicted_rd' in history and history['predicted_rd']:
        axes[2, 0].plot(history['predicted_rd'], label='Predicted Rd', linewidth=2, color='blue')  # type: ignore
        axes[2, 0].axhline(y=1.8, color='red', linestyle='--', label='Target Rd (1.8)', linewidth=2)  # type: ignore
        axes[2, 0].set_title('Rd Prediction')  # type: ignore
        axes[2, 0].legend()  # type: ignore
        axes[2, 0].grid(True, alpha=0.3)  # type: ignore
        axes[2, 0].set_ylabel('Rd Value')  # type: ignore
    
    if 'rd_loss' in history and history['rd_loss']:
        axes[2, 1].semilogy([max(x, 1e-10) for x in history['rd_loss']], label='Rd Loss', linewidth=2, color='green')  # type: ignore
        axes[2, 1].set_title('Rd Loss (Log Scale)')  # type: ignore
        axes[2, 1].legend()  # type: ignore
        axes[2, 1].grid(True, alpha=0.3)  # type: ignore
        axes[2, 1].set_ylabel('MSE Loss')  # type: ignore
    
    # Rd prediction error
    if 'predicted_rd' in history and history['predicted_rd']:
        rd_errors = [abs(pred - 1.8) for pred in history['predicted_rd']]
        axes[2, 2].plot(rd_errors, label='|Predicted - Target|', linewidth=2, color='orange')  # type: ignore
        axes[2, 2].set_title('Rd Prediction Error')  # type: ignore
        axes[2, 2].legend()  # type: ignore
        axes[2, 2].grid(True, alpha=0.3)  # type: ignore
        axes[2, 2].set_ylabel('Absolute Error')  # type: ignore
    
    plt.suptitle(f'Complete Physics Analysis - Epoch {epoch}\nCurrent Predicted Rd: {avg_predicted_rd:.4f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'complete_physics_results/epoch_{epoch:03d}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_complete_physics_model() 