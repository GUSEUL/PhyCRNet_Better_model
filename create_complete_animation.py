"""
Complete Animation Visualization
Creates GIF animations for all timesteps showing Ground Truth, Prediction, and Difference
for each field (U, V, Pressure, Temperature)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from data import MatDataset
from models import PhyCRNet

def load_trained_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Val Loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, checkpoint

def generate_all_predictions(model, dataset, device, max_timesteps=None):
    """Generate predictions for all timesteps"""
    
    print("🔮 Generating predictions for all timesteps...")
    
    # Determine number of timesteps
    total_timesteps = len(dataset)
    if max_timesteps is not None:
        total_timesteps = min(total_timesteps, max_timesteps)
    
    print(f"   Processing {total_timesteps} timesteps (Full dataset: {len(dataset)})")
    print(f"   This will create animations with {total_timesteps} frames each")
    if total_timesteps > 100:
        print(f"   ⚠️  Large number of frames - animation files will be substantial!")
    
    # Storage for results
    ground_truths = {
        'u': [], 'v': [], 'p': [], 't': []
    }
    predictions = {
        'u': [], 'v': [], 'p': [], 't': []
    }
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_timesteps), desc="Generating predictions"):
            # Get data
            input_state, target_state, ground_truth = dataset[i]
            
            # Add batch dimension and move to device
            input_state = input_state.unsqueeze(0).to(device)
            
            # Generate prediction
            prediction = model(input_state)
            
            # Convert to numpy and remove batch dimension
            pred_np = prediction.squeeze(0).cpu().numpy()
            target_np = target_state.cpu().numpy()
            
            # Store ground truth (target)
            ground_truths['u'].append(target_np[0])  # U velocity
            ground_truths['v'].append(target_np[1])  # V velocity  
            ground_truths['t'].append(target_np[2])  # Temperature
            ground_truths['p'].append(target_np[3])  # Pressure
            
            # Store prediction
            predictions['u'].append(pred_np[0])
            predictions['v'].append(pred_np[1])
            predictions['t'].append(pred_np[2])
            predictions['p'].append(pred_np[3])
    
    # Convert to numpy arrays
    for field in ['u', 'v', 'p', 't']:
        ground_truths[field] = np.array(ground_truths[field])
        predictions[field] = np.array(predictions[field])
    
    print(f"✅ Generated predictions for all timesteps")
    u_array = ground_truths['u']
    if isinstance(u_array, np.ndarray):
        print(f"   Shape: {u_array.shape}")
    
    return ground_truths, predictions

def create_field_animation(ground_truth, prediction, field_name, save_path, fps=10):
    """Create animation for a single field showing GT, Prediction, and Difference"""
    
    print(f"🎬 Creating animation for {field_name}...")
    
    # Calculate difference
    difference = prediction - ground_truth
    
    # Determine color maps and value ranges
    if field_name.lower() in ['u', 'v']:
        cmap = 'RdBu_r'
        # Use symmetric range for velocities
        vmax_gt = max(np.abs(ground_truth.min()), np.abs(ground_truth.max()))
        vmax_pred = max(np.abs(prediction.min()), np.abs(prediction.max()))
        vmax = max(vmax_gt, vmax_pred)
        vmin_gt = vmin_pred = -vmax
        vmax_gt = vmax_pred = vmax
    elif field_name.lower() == 't':
        cmap = 'hot'
        vmin_gt, vmax_gt = ground_truth.min(), ground_truth.max()
        vmin_pred, vmax_pred = prediction.min(), prediction.max()
    else:  # pressure
        cmap = 'viridis'
        vmin_gt, vmax_gt = ground_truth.min(), ground_truth.max()
        vmin_pred, vmax_pred = prediction.min(), prediction.max()
    
    # Difference always uses symmetric range
    vmax_diff = max(np.abs(difference.min()), np.abs(difference.max()))
    vmin_diff, vmax_diff = -vmax_diff, vmax_diff
    
    # Set up the figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])  # Prediction
    ax3 = fig.add_subplot(gs[0, 2])  # Difference
    
    # Initialize images
    im1 = ax1.imshow(ground_truth[0], cmap=cmap, vmin=vmin_gt, vmax=vmax_gt, 
                     aspect='equal', origin='lower')
    im2 = ax2.imshow(prediction[0], cmap=cmap, vmin=vmin_pred, vmax=vmax_pred, 
                     aspect='equal', origin='lower')
    im3 = ax3.imshow(difference[0], cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, 
                     aspect='equal', origin='lower')
    
    # Set titles
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    ax3.set_title('Difference (Pred - GT)', fontsize=14, fontweight='bold')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Remove axis ticks for cleaner look
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add main title with timestep info
    title_text = fig.suptitle(f'{field_name.upper()} Field - Timestep: 0/{len(ground_truth)-1}', 
                             fontsize=16, fontweight='bold')
    
    # Add statistics text
    stats_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    def update(frame):
        """Update function for animation"""
        
        # Update images
        im1.set_array(ground_truth[frame])
        im2.set_array(prediction[frame])
        im3.set_array(difference[frame])
        
        # Update title
        title_text.set_text(f'{field_name.upper()} Field - Timestep: {frame}/{len(ground_truth)-1}')
        
        # Calculate and update statistics
        mse = np.mean((prediction[frame] - ground_truth[frame])**2)
        mae = np.mean(np.abs(prediction[frame] - ground_truth[frame]))
        max_error = np.max(np.abs(prediction[frame] - ground_truth[frame]))
        
        stats_text.set_text(f'MSE: {mse:.6f} | MAE: {mae:.6f} | Max Error: {max_error:.6f}')
        
        return [im1, im2, im3, title_text, stats_text]
    
    # Create animation
    total_frames = len(ground_truth)
    anim = animation.FuncAnimation(fig, update, frames=total_frames, 
                                 interval=1000//fps, blit=False, repeat=True)
    
    # Save animation
    print(f"   Saving to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"✅ Animation saved: {save_path}")
    
    return save_path

def create_error_summary_plot(ground_truths, predictions, save_dir):
    """Create summary plot of errors across all timesteps"""
    
    print("📊 Creating error summary...")
    
    fields = ['u', 'v', 't', 'p']
    field_names = ['U-Velocity', 'V-Velocity', 'Temperature', 'Pressure']
    
    # Calculate errors for all fields
    errors = {}
    for field in fields:
        diff = predictions[field] - ground_truths[field]
        errors[field] = {
            'mse': np.mean(diff**2, axis=(1, 2)),  # MSE per timestep
            'mae': np.mean(np.abs(diff), axis=(1, 2)),  # MAE per timestep
            'max': np.max(np.abs(diff), axis=(1, 2))  # Max error per timestep
        }
    
    # Create summary plot
    fig, axes_2d = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes_2d.flatten()
    
    timesteps = np.arange(len(errors['u']['mse']))
    
    for i, (field, field_name) in enumerate(zip(fields, field_names)):
        ax = axes[i]
        
        # Plot error metrics
        ax.plot(timesteps, errors[field]['mse'], 'b-', label='MSE', linewidth=2, alpha=0.7)
        ax.plot(timesteps, errors[field]['mae'], 'r-', label='MAE', linewidth=2, alpha=0.7)
        ax.plot(timesteps, errors[field]['max'], 'g-', label='Max Error', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{field_name} - Error Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
    
    plt.suptitle('Prediction Error Summary Across All Timesteps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'error_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Error summary saved: {summary_path}")
    
    # Print overall statistics
    print(f"\n📈 Overall Statistics:")
    for field, field_name in zip(fields, field_names):
        avg_mse = np.mean(errors[field]['mse'])
        avg_mae = np.mean(errors[field]['mae'])
        avg_max = np.mean(errors[field]['max'])
        print(f"   {field_name:12}: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, Max={avg_max:.6f}")
    
    return summary_path

def main():
    """Main function to create all animations"""
    
    print("🎬 Complete Animation Creator")
    print("=" * 60)
    
    # Configuration
    config = {
        'checkpoint_path': 'complete_physics_model_checkpoint.pth',  # Update as needed
        'data_file': 'Ra_10^5_Rd_1.8.mat',
        'output_dir': 'complete_animations',
        'max_timesteps': None,  # Use ALL timesteps (299)
        'fps': 10,  # Frames per second for smoother animation
    }
    
    # Alternative checkpoint paths to try
    alternative_checkpoints = [
        'complete_physics_model_checkpoint.pth',
        'physics_enhanced_model_checkpoint.pth', 
        'safe_physics_model_checkpoint.pth',
        'clean_model_checkpoint.pth'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Try to load model
    model = None
    checkpoint = None
    
    for checkpoint_path in alternative_checkpoints:
        if os.path.exists(checkpoint_path):
            try:
                model, checkpoint = load_trained_model(checkpoint_path, device)
                config['checkpoint_path'] = checkpoint_path
                break
            except Exception as e:
                print(f"⚠️ Failed to load {checkpoint_path}: {e}")
                continue
    
    if model is None:
        print("❌ No valid checkpoint found! Available files:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"   - {file}")
        return
    
    # Load dataset
    print(f"\n📁 Loading dataset: {config['data_file']}")
    dataset = MatDataset(config['data_file'])
    print(f"   Total timesteps available: {len(dataset)}")
    
    # Generate predictions
    ground_truths, predictions = generate_all_predictions(
        model, dataset, device, config['max_timesteps']
    )
    
    # Create animations for each field
    fields = ['u', 'v', 't', 'p']
    field_names = ['U-Velocity', 'V-Velocity', 'Temperature', 'Pressure']
    
    animation_paths = []
    
    for field, field_name in zip(fields, field_names):
        save_path = os.path.join(config['output_dir'], f'{field}_field_animation.gif')
        
        anim_path = create_field_animation(
            ground_truths[field], 
            predictions[field],
            field_name,
            save_path,
            fps=config['fps']
        )
        animation_paths.append(anim_path)
    
    # Create error summary
    summary_path = create_error_summary_plot(ground_truths, predictions, config['output_dir'])
    
    # Final summary
    print(f"\n🎉 All animations created successfully!")
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"📝 Generated files:")
    for path in animation_paths:
        print(f"   - {os.path.basename(path)}")
    print(f"   - {os.path.basename(summary_path)}")
    
    print(f"\n⚡ Animation Details:")
    print(f"   - Timesteps: {len(ground_truths['u'])} (ALL available timesteps)")
    print(f"   - FPS: {config['fps']}")
    print(f"   - Duration: ~{len(ground_truths['u'])/config['fps']:.1f} seconds each")
    print(f"   - Fields: U, V, Temperature, Pressure")
    print(f"   - Shows: Ground Truth | Prediction | Difference")
    print(f"   - Total frames per animation: {len(ground_truths['u'])}")
    print(f"\n🎬 Each GIF shows the complete temporal evolution of the field!")
    print(f"   You can observe how the model predictions compare to ground truth")
    print(f"   across the entire simulation timespan.")

if __name__ == "__main__":
    main() 