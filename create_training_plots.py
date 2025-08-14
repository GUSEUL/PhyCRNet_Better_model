"""
Training Metrics Visualization
Creates comprehensive plots for MSE, max error, Rd loss, PDE loss, and data loss
from trained model checkpoint and test set evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from data import MatDataset
from models import PhyCRNet
from accurate_physics_loss import AccuratePhysicsLoss

def load_checkpoint_history(checkpoint_path):
    """Load training history from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'history' not in checkpoint:
        raise ValueError("No training history found in checkpoint")
    
    history = checkpoint['history']
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Available history keys: {list(history.keys())}")
    
    return history, checkpoint

def load_trained_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to determine model architecture
    model_keys = list(checkpoint['model_state_dict'].keys())
    has_ra_head = any('ra_head' in key for key in model_keys)
    
    # Initialize model
    try:
        model = PhyCRNet(ch=4, hidden=192, dropout_rate=0.2, predict_ra=has_ra_head).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e1:
        try:
            model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1, predict_ra=has_ra_head).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e2:
            model = PhyCRNet(ch=4, hidden=192, dropout_rate=0.2, predict_ra=False).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print(f"Model loaded successfully")
    
    return model, checkpoint

def evaluate_test_metrics(model, dataset, device, max_samples=None):
    """Evaluate MSE and max error on test set"""
    
    print("Evaluating test set metrics...")
    
    # Create test split (same as training)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_indices = list(range(train_size + val_size, total_size))
    
    if max_samples is not None:
        test_indices = test_indices[:max_samples]
    
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Testing on {len(test_indices)} samples")
    
    # Storage for metrics
    mse_values = {'u': [], 'v': [], 't': [], 'p': []}
    max_error_values = {'u': [], 'v': [], 't': [], 'p': []}
    ra_predictions = []
    
    # Get dataset parameters for physics loss
    dataset_params = dataset.get_params()
    nanofluid_props = dataset.get_nanofluid_properties()
    
    # Initialize physics loss
    physics_loss_fn = AccuratePhysicsLoss(
        dataset_params, nanofluid_props, 
        dt=dataset_params.get('dt', 0.0001)
    ).to(device)
    
    physics_losses = []
    
    model.eval()
    with torch.no_grad():
        for input_state, target_state, _ in tqdm(test_loader, desc="Evaluating"):
            input_state = input_state.to(device)
            target_state = target_state.to(device)
            
            # Generate prediction
            model_output = model(input_state)
            
            if isinstance(model_output, tuple):
                prediction, ra_scalar = model_output
                prediction = prediction[:, :4]  # Only main fields
                # Convert log10(Ra) back to Ra
                ra_predictions.extend(torch.pow(10, ra_scalar).cpu().numpy())
            else:
                prediction = model_output
            
            # Calculate physics loss (use input and prediction as temporal pair)
            try:
                physics_result = physics_loss_fn(input_state, prediction, validation_mode=True)
                if isinstance(physics_result, dict) and 'total' in physics_result:
                    # Use the 'total' field directly
                    total_physics = physics_result['total']
                elif isinstance(physics_result, dict):
                    # Sum up only the physics loss components (skip 'residuals' dict)
                    total_physics = 0.0
                    for key, component_loss in physics_result.items():
                        if key != 'residuals' and torch.is_tensor(component_loss):
                            total_physics += component_loss
                        elif key != 'residuals' and isinstance(component_loss, (int, float)):
                            total_physics += component_loss
                elif torch.is_tensor(physics_result):
                    total_physics = physics_result
                else:
                    total_physics = torch.tensor(physics_result)
                
                # Ensure we have a scalar value
                if torch.is_tensor(total_physics):
                    physics_losses.append(float(total_physics.cpu().item()))
                else:
                    physics_losses.append(float(total_physics))
            except Exception as e:
                # Fallback: skip physics loss calculation if it fails
                print(f"Warning: Physics loss calculation failed: {e}")
                physics_losses.append(0.0)
            
            # Calculate field-wise metrics
            pred_np = prediction.cpu().numpy()
            target_np = target_state.cpu().numpy()
            
            fields = ['u', 'v', 't', 'p']
            for i, field in enumerate(fields):
                # MSE per sample
                mse_per_sample = np.mean((pred_np[:, i] - target_np[:, i])**2, axis=(1, 2))
                mse_values[field].extend(mse_per_sample)
                
                # Max error per sample
                max_error_per_sample = np.max(np.abs(pred_np[:, i] - target_np[:, i]), axis=(1, 2))
                max_error_values[field].extend(max_error_per_sample)
    
    # Convert to numpy arrays
    for field in fields:
        mse_values[field] = np.array(mse_values[field])
        max_error_values[field] = np.array(max_error_values[field])
    
    physics_losses = np.array(physics_losses)
    ra_predictions = np.array(ra_predictions) if ra_predictions else None
    
    print(f"Test evaluation completed")
    print(f"  Average MSE - U: {np.mean(mse_values['u']):.6f}, V: {np.mean(mse_values['v']):.6f}")
    print(f"  Average MSE - T: {np.mean(mse_values['t']):.6f}, P: {np.mean(mse_values['p']):.6f}")
    print(f"  Average Physics Loss: {np.mean(physics_losses):.6f}")
    
    return mse_values, max_error_values, physics_losses, ra_predictions, dataset_params

def create_comprehensive_plots(history, mse_values, max_error_values, physics_losses, ra_predictions, dataset_params, save_dir):
    """Create comprehensive training and evaluation plots"""
    
    print("Creating comprehensive plots...")
    
    # Create figure with 5 subplots arranged in 2 rows
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training Loss Evolution (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(len(history['train_losses']))
    ax1.semilogy(epochs, history['train_losses'], 'b-', linewidth=2, alpha=0.8, label='Train Loss')
    ax1.semilogy(epochs, history['val_losses'], 'r-', linewidth=2, alpha=0.8, label='Val Loss')
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Data Loss vs Physics Loss (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'data_losses' in history and 'unweighted_physics' in history:
        ax2.semilogy(epochs, history['data_losses'], 'g-', linewidth=2, alpha=0.8, label='Data Loss')
        ax2.semilogy(epochs, history['unweighted_physics'], 'orange', linewidth=2, alpha=0.8, label='Physics Loss')
        ax2.set_title('Data vs Physics Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Data/Physics Loss\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Data vs Physics Loss', fontsize=14, fontweight='bold')
    
    # 3. Ra Loss Evolution (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'ra_loss' in history:
        ax3.semilogy([max(x, 1e-10) for x in history['ra_loss']], 'green', linewidth=2, alpha=0.8)
        ax3.set_title('Ra Loss Evolution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Ra Loss (Log Scale)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Ra Loss\nNot Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Ra Loss Evolution', fontsize=14, fontweight='bold')
    
    # 4. Average Test MSE by Field (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    fields = ['u', 'v', 't', 'p']
    field_names = ['U-Velocity', 'V-Velocity', 'Temperature', 'Pressure']
    colors = ['blue', 'red', 'green', 'orange']
    
    avg_mse = [np.mean(mse_values[field]) for field in fields]
    bars = ax4.bar(field_names, avg_mse, color=colors, alpha=0.7)
    ax4.set_title('Average Test MSE by Field', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Field')
    ax4.set_ylabel('Average MSE')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_mse):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 5. Average Test Max Error by Field (Bottom Center)
    ax5 = fig.add_subplot(gs[1, 1])
    avg_max_error = [np.mean(max_error_values[field]) for field in fields]
    bars = ax5.bar(field_names, avg_max_error, color=colors, alpha=0.7)
    ax5.set_title('Average Test Max Error by Field', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Field')
    ax5.set_ylabel('Average Max Error')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_max_error):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=10)

    
    plt.suptitle('Comprehensive Training and Test Analysis', fontsize=18, fontweight='bold')
    
    # Save plot
    save_path = os.path.join(save_dir, 'comprehensive_training_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plot saved: {save_path}")
    
    return save_path

def main():
    """Main function to create training plots"""
    
    print("Training Metrics Visualization Creator")
    print("=" * 60)
    
    # Configuration
    config = {
        'checkpoint_path': 'complete_physics_model_Da_0.100_checkpoint.pth',
        'data_file': '01_Da_0.100.mat',
        'output_dir': 'training_plots',
        'max_test_samples': 200,  # Limit test samples for faster evaluation
    }
    
    # Alternative checkpoint paths
    alternative_checkpoints = [
        'complete_physics_model_Da_0.100_checkpoint.pth',
        'complete_physics_model_checkpoint.pth',
        'physics_enhanced_model_checkpoint.pth'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load checkpoint and history
    checkpoint_path = None
    for path in alternative_checkpoints:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("No checkpoint found!")
        return
    
    try:
        history, checkpoint = load_checkpoint_history(checkpoint_path)
        model, _ = load_trained_model(checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # Load dataset
    if not os.path.exists(config['data_file']):
        print(f"Data file {config['data_file']} not found!")
        return
    
    dataset = MatDataset(config['data_file'])
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Evaluate test metrics
    try:
        mse_values, max_error_values, physics_losses, ra_predictions, dataset_params = evaluate_test_metrics(
            model, dataset, device, config['max_test_samples']
        )
    except Exception as e:
        print(f"Test evaluation failed: {e}")
        return
    
    # Create plots
    try:
        plot_path = create_comprehensive_plots(
            history, mse_values, max_error_values, physics_losses, ra_predictions, 
            dataset_params, config['output_dir']
        )
        
        print(f"\nTraining analysis completed successfully!")
        print(f"Output directory: {config['output_dir']}")
        print(f"Generated plot: {os.path.basename(plot_path)}")
        
    except Exception as e:
        print(f"Plot creation failed: {e}")
        return

if __name__ == "__main__":
    main()