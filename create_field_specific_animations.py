"""
Field-Specific Comparison Animations
Create animations comparing Ground Truth vs Prediction vs Difference for each field
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from data import MatDataset
from models import PhyCRNet
from tqdm import tqdm

def load_trained_model(checkpoint_path, device):
    """Load trained model"""
    print(f"Loading trained model from {checkpoint_path}...")
    
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'Unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'Unknown')
        
        print(f"Model loaded successfully!")
        print(f"   Training epoch: {epoch}")
        print(f"   Best validation loss: {best_val_loss}")
        
        model.eval()
        return model, checkpoint
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def extract_sequences(dataset, model, device, num_samples=50):
    """Extract sequences"""
    print(f"Extracting {num_samples} sequences for comparison...")
    
    total_samples = len(dataset)
    indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
    
    ground_truths = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(indices, desc="Extracting samples"):
            input_state, target_state, _ = dataset[i]
            
            # Perform prediction
            input_batch = input_state.unsqueeze(0).to(device)
            prediction = model(input_batch)
            
            # Save
            ground_truths.append(target_state.numpy())
            predictions.append(prediction.cpu().numpy()[0])
    
    return np.array(ground_truths), np.array(predictions), indices

def create_field_animation(ground_truths, predictions, field_idx, field_name, cmap, save_path, fps=10):
    """Create GT vs Pred vs Diff animation for individual field"""
    
    print(f"Creating {field_name} animation...")
    
    # Extract data
    gt_field = ground_truths[:, field_idx, :, :]  # [num_samples, H, W]
    pred_field = predictions[:, field_idx, :, :]
    diff_field = gt_field - pred_field
    
    # Calculate colormap range
    # GT and Pred use same range
    combined_data = np.concatenate([gt_field.flatten(), pred_field.flatten()])
    vmin, vmax = np.percentile(combined_data, [1, 99])
    
    # Velocity fields use symmetric range
    if field_idx < 2:  # U, V velocity
        max_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -max_abs, max_abs
    
    # Difference uses separate range
    diff_max = np.percentile(np.abs(diff_field), 95)
    diff_vmin, diff_vmax = -diff_max, diff_max
    
    print(f"   Data range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"   Diff range: [{diff_vmin:.4f}, {diff_vmax:.4f}]")
    
    # Figure setup (1x3 layout)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{field_name} Comparison: Ground Truth | Prediction | Difference', 
                 fontsize=16, weight='bold')
    
    # Initial image setup
    im1 = axes[0].imshow(gt_field[0], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect='equal', origin='lower', interpolation='bilinear')
    axes[0].set_title('Ground Truth', fontsize=14, weight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    im2 = axes[1].imshow(pred_field[0], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect='equal', origin='lower', interpolation='bilinear')
    axes[1].set_title('Prediction', fontsize=14, weight='bold')
    axes[1].set_xlabel('X')
    
    im3 = axes[2].imshow(diff_field[0], cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax,
                        aspect='equal', origin='lower', interpolation='bilinear')
    axes[2].set_title('Difference (GT - Pred)', fontsize=14, weight='bold')
    axes[2].set_xlabel('X')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.02)
    cbar1.set_label(field_name, fontsize=12)
    
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
    cbar2.set_label(field_name, fontsize=12)
    
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.02)
    cbar3.set_label('Difference', fontsize=12)
    
    # Axis settings
    for ax in axes:
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.tick_params(labelsize=10)
    
    # Info text
    info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=14, weight='bold',
                        bbox=dict(boxstyle="round", facecolor='lightcyan', alpha=0.9))
    
    def update_frame(frame):
        """Frame update function"""
        # Update images
        im1.set_array(gt_field[frame])
        im2.set_array(pred_field[frame])
        im3.set_array(diff_field[frame])
        
        # Calculate statistics
        mse = np.mean((gt_field[frame] - pred_field[frame])**2)
        mae = np.mean(np.abs(gt_field[frame] - pred_field[frame]))
        correlation = np.corrcoef(gt_field[frame].flatten(), pred_field[frame].flatten())[0, 1]
        
        # Value ranges
        gt_min, gt_max = gt_field[frame].min(), gt_field[frame].max()
        pred_min, pred_max = pred_field[frame].min(), pred_field[frame].max()
        diff_min, diff_max = diff_field[frame].min(), diff_field[frame].max()
        
        # Update info
        progress = (frame + 1) / len(gt_field) * 100
        info_text.set_text(
            f'Frame: {frame+1}/{len(gt_field)} ({progress:.1f}%) | '
            f'MSE: {mse:.6f} | MAE: {mae:.6f} | Correlation: {correlation:.4f} | '
            f'GT Range: [{gt_min:.3f}, {gt_max:.3f}] | '
            f'Pred Range: [{pred_min:.3f}, {pred_max:.3f}] | '
            f'Diff Range: [{diff_min:.3f}, {diff_max:.3f}]'
        )
        
        return [im1, im2, im3, info_text]
    
    # Create animation
    print(f"   Frames: {len(gt_field)}")
    print(f"   Duration: {len(gt_field)/fps:.1f} seconds at {fps} FPS")
    
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(gt_field),
        interval=1000//fps, blit=True, repeat=True
    )
    
    # Save
    print(f"   Saving to {save_path}")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100,
              savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
    plt.close()
    
    return anim

def calculate_field_accuracy(ground_truths, predictions, field_idx, field_name):
    """Calculate field-specific accuracy"""
    gt_field = ground_truths[:, field_idx, :, :]
    pred_field = predictions[:, field_idx, :, :]
    
    mse = np.mean((gt_field - pred_field)**2)
    mae = np.mean(np.abs(gt_field - pred_field))
    correlation = np.corrcoef(gt_field.flatten(), pred_field.flatten())[0, 1]
    
    # Relative error (MAPE)
    # Be careful due to values close to zero
    nonzero_mask = np.abs(gt_field) > 1e-6
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((gt_field[nonzero_mask] - pred_field[nonzero_mask]) / gt_field[nonzero_mask])) * 100
    else:
        mape = float('inf')
    
    # Maximum absolute error
    max_error = np.max(np.abs(gt_field - pred_field))
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'mape': mape,
        'max_error': max_error,
        'field_name': field_name
    }

def main():
    """Main function"""
    print("Field-Specific Comparison Animations Creator")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_checkpoint = 'clean_model_checkpoint.pth'
    output_dir = 'field_specific_animations'
    num_samples = 60  # Animation frame count
    
    # Field information
    field_configs = [
        {'idx': 0, 'name': 'U-velocity', 'cmap': 'RdBu_r', 'filename': 'u_velocity_comparison.gif'},
        {'idx': 1, 'name': 'V-velocity', 'cmap': 'RdBu_r', 'filename': 'v_velocity_comparison.gif'},
        {'idx': 2, 'name': 'Temperature', 'cmap': 'hot', 'filename': 'temperature_comparison.gif'},
        {'idx': 3, 'name': 'Pressure', 'cmap': 'viridis', 'filename': 'pressure_comparison.gif'}
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    model, checkpoint = load_trained_model(model_checkpoint, device)
    if model is None:
        print("Failed to load trained model. Please train the model first.")
        print("   Run: python train_simple_clean.py")
        return
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = MatDataset('Ra_10^5_Rd_1.8.mat', device='cpu')
        print(f"Dataset loaded: {len(dataset)} time steps available")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Extract sequences
    ground_truths, predictions, indices = extract_sequences(dataset, model, device, num_samples)
    
    print(f"Sequences extracted:")
    print(f"   GT shape: {ground_truths.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    
    # Create field-specific animations
    print(f"\nCreating field-specific animations...")
    accuracy_results = []
    
    for config in field_configs:
        field_idx = config['idx']
        field_name = config['name']
        cmap = config['cmap']
        filename = config['filename']
        
        print(f"\n{'='*50}")
        print(f"Processing {field_name}...")
        
        # Create animation
        save_path = os.path.join(output_dir, filename)
        create_field_animation(
            ground_truths, predictions, field_idx, field_name, cmap, save_path, fps=10
        )
        
        # Calculate accuracy
        accuracy = calculate_field_accuracy(ground_truths, predictions, field_idx, field_name)
        accuracy_results.append(accuracy)
        
        print(f"{field_name} animation completed!")
        print(f"   MSE: {accuracy['mse']:.6f}")
        print(f"   Correlation: {accuracy['correlation']:.4f}")
    
    # Overall results summary
    print(f"\nAll field-specific animations created!")
    print(f"Check '{output_dir}' directory for results")
    print(f"\nCreated animations:")
    for config in field_configs:
        print(f"  • {config['name']}: {config['filename']}")
    
    # Accuracy report
    print(f"\nField-Specific Accuracy Report:")
    print("=" * 70)
    print(f"{'Field':12} {'MSE':>10} {'MAE':>10} {'Correlation':>12} {'MAPE(%)':>10} {'Max Error':>12}")
    print("-" * 70)
    
    for result in accuracy_results:
        mape_str = f"{result['mape']:.2f}" if result['mape'] != float('inf') else "N/A"
        print(f"{result['field_name']:12} {result['mse']:>10.6f} {result['mae']:>10.6f} "
              f"{result['correlation']:>12.4f} {mape_str:>10} {result['max_error']:>12.6f}")
    
    # Overall statistics
    total_mse = np.mean([r['mse'] for r in accuracy_results])
    total_correlation = np.mean([r['correlation'] for r in accuracy_results])
    
    print("-" * 70)
    print(f"{'AVERAGE':12} {total_mse:>10.6f} {'-':>10} {total_correlation:>12.4f} {'-':>10} {'-':>12}")
    
    # Training information
    if checkpoint:
        print(f"\nModel Training Info:")
        print(f"   Final epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'Unknown')}")

if __name__ == "__main__":
    main() 