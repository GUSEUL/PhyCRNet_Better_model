"""
Training Results Animation Creator
Create animations from existing training results
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

def create_training_evolution_video(results_dir='clean_training_results', output_path='training_evolution.mp4'):
    """Create a video showing training evolution"""
    
    prediction_files, loss_files, epochs = load_training_images(results_dir)
    
    if prediction_files is None:
        print("No training images found!")
        return
    
    print(f"Creating training evolution video with {len(prediction_files)} frames...")
    
    # Create figure for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def update_frame(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Load and display prediction image
        if frame_idx < len(prediction_files):
            pred_img = Image.open(prediction_files[frame_idx])
            ax1.imshow(pred_img)
            ax1.set_title(f'Epoch {epochs[frame_idx]} - Model Predictions')
            ax1.axis('off')
        
        # Load and display loss plot
        if frame_idx < len(loss_files):
            loss_img = Image.open(loss_files[frame_idx])
            ax2.imshow(loss_img)
            ax2.set_title(f'Epoch {epochs[frame_idx]} - Training Loss')
            ax2.axis('off')
        
        plt.tight_layout()
    
    # Create animation
    anim = animation.FuncAnimation(fig, update_frame, frames=len(prediction_files), 
                                 interval=1000, repeat=True)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=2, dpi=100)
    plt.close(fig)
    
    print(f"Training evolution video saved as: {output_path}")

def main():
    """Main function to create training animations"""
    
    print("PhyCRNet Training Animation Creator")
    print("=" * 50)
    
    # Check if training results exist
    results_dir = 'clean_training_results'
    if not os.path.exists(results_dir):
        print(f"Training results directory '{results_dir}' not found!")
        print("Please run train_simple_clean.py first to generate training results.")
        return
    
    try:
        # Create training evolution video
        create_training_evolution_video(results_dir)
        
        print("\nAnimation creation completed!")
        print("Check the generated .mp4 files for training visualization.")
        
    except Exception as e:
        print(f"Error creating animations: {e}")
        print("Make sure you have ffmpeg installed for video creation.")

if __name__ == "__main__":
    main() 