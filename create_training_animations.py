"""
Training Results Animation Creator
Create animations from existing training results (no new predictions needed)
Uses saved images from train_simple_clean.py results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
from PIL import Image
import torch

def load_training_images(results_dir='clean_training_results'):
    """Load saved training result images"""
    print(f"Loading training images from {results_dir}...")
    
    # Find all prediction images
    prediction_files = glob.glob(os.path.join(results_dir, 'epoch_*_prediction.png'))
    loss_files = glob.glob(os.path.join(results_dir, 'epoch_*_losses.png'))
    
    if not prediction_files:
        print(f"No prediction images found in {results_dir}")
        return None, None
    
    # Sort by epoch number
    prediction_files.sort(key=lambda x: int(x.split('epoch_')[1].split('_')[0]))
    loss_files.sort(key=lambda x: int(x.split('epoch_')[1].split('_')[0]))
    
    print(f"Found {len(prediction_files)} prediction images")
    print(f"Found {len(loss_files)} loss plots")
    
    epochs = [int(f.split('epoch_')[1].split('_')[0]) for f in prediction_files]
    
    return prediction_files, loss_files, epochs

def extract_fields_from_training_images(image_paths):
    """Extract field data from saved training prediction images"""
    print("Extracting field data from training images...")
    
    # We'll read the images and try to extract the actual field data
    # This is a simplified approach - in practice, it's better to save raw data
    
    inputs = []
    targets = []
    predictions = []
    
    for img_path in image_paths:
        # Load the image
        img = Image.open(img_path)
        
        # For this simplified version, we'll just store the image paths
        # In a real implementation, you'd want to save the raw numpy arrays
        inputs.append(img_path)
        targets.append(img_path)  
        predictions.append(img_path)
    
    return inputs, targets, predictions

def load_saved_results_from_checkpoint(checkpoint_path, data_file, device, use_all_data=False):
    """Load model and generate results for all available time steps"""
    print(f"Loading results from checkpoint: {checkpoint_path}")
    
    # Load model
    from models import PhyCRNet
    model = PhyCRNet(ch=4, hidden=128, dropout_rate=0.1).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown')}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None, None
    
    # Load dataset
    from data import MatDataset
    dataset = MatDataset(data_file, device='cpu')
    
    total_size = len(dataset)
    print(f"Total dataset size: {total_size} time steps")
    
    if use_all_data:
        # Use all available data
        indices = list(range(total_size))
        print(f"Using ALL {total_size} time steps for animation")
    else:
        # Use validation split (same as in training) 
        train_size = int(0.8 * total_size)
        indices = list(range(train_size, total_size))
        print(f"Using validation split: {len(indices)} time steps (from {train_size} to {total_size-1})")
    
    inputs = []
    targets = []
    predictions = []
    
    print(f"Generating predictions for {len(indices)} time steps...")
    print("This may take a while for large datasets...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if (i + 1) % 100 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{len(indices)} samples...")
            
            input_state, target_state, _ = dataset[idx]
            
            # Generate prediction
            input_batch = input_state.unsqueeze(0).to(device)
            prediction = model(input_batch)
            
            inputs.append(input_state.numpy())
            targets.append(target_state.numpy())
            predictions.append(prediction.cpu().numpy()[0])
    
    print(f"Completed generating {len(predictions)} predictions!")
    return np.array(inputs), np.array(targets), np.array(predictions), indices

def create_simple_comparison_animation(inputs, targets, predictions, save_path, fps=5):
    """Create a simplified comparison animation"""
    
    print("Creating comparison animation...")
    
    differences = targets - predictions
    
    # Field information
    field_names = ['U-velocity', 'V-velocity', 'Temperature', 'Pressure']
    cmaps = ['RdBu_r', 'RdBu_r', 'hot', 'viridis']
    
    # Calculate value ranges
    value_ranges = []
    diff_ranges = []
    
    for field_idx in range(4):
        all_data = np.concatenate([
            targets[:, field_idx].flatten(),
            predictions[:, field_idx].flatten()
        ])
        vmin, vmax = np.percentile(all_data, [2, 98])
        
        if field_idx < 2:  # Velocity fields - symmetric
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs
        
        value_ranges.append((vmin, vmax))
        
        # Difference range
        diff_data = differences[:, field_idx]
        diff_max = np.percentile(np.abs(diff_data), 90)
        diff_ranges.append((-diff_max, diff_max))
    
    # Create figure (3x4: GT, Pred, Diff for each field)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Training Results: Ground Truth | Prediction | Difference', 
                 fontsize=14, weight='bold')
    
    # Initialize images
    images = []
    
    for field_idx in range(4):
        vmin, vmax = value_ranges[field_idx]
        diff_vmin, diff_vmax = diff_ranges[field_idx]
        cmap = cmaps[field_idx]
        field_name = field_names[field_idx]
        
        # Ground Truth
        im_gt = axes[0, field_idx].imshow(targets[0, field_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                                        aspect='equal', origin='lower')
        axes[0, field_idx].set_title(f'{field_name}\n(Ground Truth)', fontsize=10)
        
        # Prediction
        im_pred = axes[1, field_idx].imshow(predictions[0, field_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                                          aspect='equal', origin='lower')
        axes[1, field_idx].set_title(f'{field_name}\n(Prediction)', fontsize=10)
        
        # Difference
        im_diff = axes[2, field_idx].imshow(differences[0, field_idx], cmap='RdBu_r',
                                          vmin=diff_vmin, vmax=diff_vmax, aspect='equal', origin='lower')
        axes[2, field_idx].set_title(f'{field_name}\n(GT - Pred)', fontsize=10)
        
        images.append([im_gt, im_pred, im_diff])
        
        # Add colorbars
        plt.colorbar(im_gt, ax=axes[0, field_idx], shrink=0.7)
        plt.colorbar(im_pred, ax=axes[1, field_idx], shrink=0.7)
        plt.colorbar(im_diff, ax=axes[2, field_idx], shrink=0.7)
    
    # Remove ticks
    for i in range(3):
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    # Add info text
    info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, weight='bold',
                        bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    
    def update_frame(frame):
        """Update animation frame"""
        for field_idx in range(4):
            images[field_idx][0].set_array(targets[frame, field_idx])      # GT
            images[field_idx][1].set_array(predictions[frame, field_idx])  # Pred
            images[field_idx][2].set_array(differences[frame, field_idx])  # Diff
        
        # Calculate metrics
        mse = np.mean((targets[frame] - predictions[frame])**2)
        mae = np.mean(np.abs(targets[frame] - predictions[frame]))
        
        # Calculate correlation
        corr = np.corrcoef(targets[frame].flatten(), predictions[frame].flatten())[0, 1]
        
        info_text.set_text(
            f'Frame: {frame+1}/{len(targets)} | MSE: {mse:.6f} | MAE: {mae:.6f} | Correlation: {corr:.4f}'
        )
        
        updated = [info_text]
        for field_images in images:
            updated.extend(field_images)
        return updated
    
    print(f"   Creating animation with {len(targets)} frames at {fps} FPS")
    
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(targets),
        interval=1000//fps, blit=True, repeat=True
    )
    
    print(f"   Saving to {save_path}")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100,
              savefig_kwargs={'bbox_inches': 'tight'})
    plt.close()
    
    return anim

def calculate_accuracy_summary(targets, predictions):
    """Calculate and display accuracy summary"""
    field_names = ['U-velocity', 'V-velocity', 'Temperature', 'Pressure']
    
    print("\nAccuracy Summary:")
    print("=" * 60)
    print(f"{'Field':12} {'MSE':>10} {'MAE':>10} {'Correlation':>12}")
    print("-" * 60)
    
    overall_metrics = []
    
    for field_idx, field_name in enumerate(field_names):
        gt_field = targets[:, field_idx, :, :]
        pred_field = predictions[:, field_idx, :, :]
        
        mse = np.mean((gt_field - pred_field)**2)
        mae = np.mean(np.abs(gt_field - pred_field))
        correlation = np.corrcoef(gt_field.flatten(), pred_field.flatten())[0, 1]
        
        print(f"{field_name:12} {mse:>10.6f} {mae:>10.6f} {correlation:>12.4f}")
        overall_metrics.append([mse, mae, correlation])
    
    # Overall averages
    avg_metrics = np.mean(overall_metrics, axis=0)
    print("-" * 60)
    print(f"{'AVERAGE':12} {avg_metrics[0]:>10.6f} {avg_metrics[1]:>10.6f} {avg_metrics[2]:>12.4f}")

def main():
    """Main function"""
    print("Training Results Animation Creator")
    print("Using existing training results - no new training needed")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'clean_model_checkpoint.pth'
    data_file = 'Ra_10^5_Rd_1.8.mat'
    output_dir = 'training_animations'
    results_dir = 'clean_training_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if training results exist
    if not os.path.exists(results_dir):
        print(f"Training results directory '{results_dir}' not found.")
        print("Please run 'python train_simple_clean.py' first to generate training results.")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint '{checkpoint_path}' not found.")
        print("Please run 'python train_simple_clean.py' first to train the model.")
        return
    
    # Method 1: Use saved training images to show training progress
    prediction_files, loss_files, epochs = load_training_images(results_dir)
    
    if prediction_files:
        print(f"\nFound training results from epochs: {epochs}")
        print("Creating training progress summary...")
        
        # Create a simple training progress animation using saved images
        print("Note: For detailed field analysis, using checkpoint method...")
    
    # Method 2: Load checkpoint and generate results for all available time steps
    print(f"\nLoading model from checkpoint for all available time steps...")
    
    # Ask user preference or use all data
    use_all_data = True  # Set to True to use ALL time steps, False for validation only
    
    inputs, targets, predictions, indices = load_saved_results_from_checkpoint(
        checkpoint_path, data_file, device, use_all_data=use_all_data
    )
    
    if inputs is not None:
        num_steps = len(predictions)
        print(f"Generated predictions for {num_steps} time steps: {indices[0]} to {indices[-1]}")
        
        # Adjust FPS based on number of frames for reasonable duration
        if num_steps <= 100:
            fps = 8
        elif num_steps <= 500:
            fps = 12
        elif num_steps <= 1000:
            fps = 15
        else:
            fps = 20  # For very long sequences
        
        # Create main animation
        animation_filename = f'all_timesteps_{num_steps}frames.gif'
        animation_path = os.path.join(output_dir, animation_filename)
        create_simple_comparison_animation(inputs, targets, predictions, animation_path, fps=fps)
        
        # Calculate accuracy
        calculate_accuracy_summary(targets, predictions)
        
        print(f"\nAnimation saved to: {animation_path}")
        print(f"Duration: {len(targets)/fps:.1f} seconds at {fps} FPS")
        print(f"Total frames: {num_steps}")
    
    print(f"\nResults available in '{output_dir}' directory")
    print("This animation shows validation results from your trained model!")

if __name__ == "__main__":
    main() 