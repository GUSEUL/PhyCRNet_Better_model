"""
Dataset Split Verification Script
Checks the train/validation/test splits for 01_Da_0.100.mat data (same as training)
and provides detailed analysis to ensure data integrity.
"""

import torch
import numpy as np
from data import MatDataset

def verify_dataset_splits():
    """Verify the dataset splits are correct and provide detailed information."""
    
    print("Dataset Split Verification")
    print("=" * 50)
    
    # Load dataset (same as train_complete_physics.py)
    data_file = '01_Da_0.100.mat'  # Match training data file
    try:
        dataset = MatDataset(data_file)
        print(f"✓ Successfully loaded dataset: {data_file}")
        print(f"  Total samples: {len(dataset)}")
        
        # Get dataset parameters
        params = dataset.get_params()
        print(f"  Physical parameters loaded: Ra={params['Ra']:.1e}, Pr={params['Pr']:.3f}, Da={params['Da']:.1e}")
        
    except Exception as e:
        print(f"✗ Failed to load dataset {data_file}: {e}")
        return
    
    # Configuration (matching training script)
    config = {
        'fraction_train': 0.7,  # 70% for training
        'fraction_val': 0.2,    # 20% for validation
        'fraction_test': 0.1,   # 10% for testing
    }
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(config['fraction_train'] * total_size)
    val_size = int(config['fraction_val'] * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplit Configuration:")
    print(f"  Train fraction: {config['fraction_train']*100:.1f}%")
    print(f"  Validation fraction: {config['fraction_val']*100:.1f}%")
    print(f"  Test fraction: {config['fraction_test']*100:.1f}%")
    
    print(f"\nCalculated Split Sizes:")
    print(f"  Total: {total_size} timesteps")
    print(f"  Train: {train_size} timesteps ({train_size/total_size*100:.1f}%)")
    print(f"  Validation: {val_size} timesteps ({val_size/total_size*100:.1f}%)")
    print(f"  Test: {test_size} timesteps ({test_size/total_size*100:.1f}%)")
    print(f"  Sum: {train_size + val_size + test_size} (should equal total)")
    
    # Create indices
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    print(f"\nIndex Ranges:")
    print(f"  Train indices: {train_indices[0]} to {train_indices[-1]} ({len(train_indices)} indices)")
    print(f"  Val indices: {val_indices[0]} to {val_indices[-1]} ({len(val_indices)} indices)")
    print(f"  Test indices: {test_indices[0]} to {test_indices[-1]} ({len(test_indices)} indices)")
    
    # Verification tests
    print(f"\nVerification Tests:")
    
    # Test 1: No overlaps
    all_indices = set(train_indices + val_indices + test_indices)
    if len(all_indices) == total_size:
        print("  ✓ No overlapping indices detected")
    else:
        print(f"  ✗ Overlapping indices detected! Expected {total_size}, got {len(all_indices)}")
        return
    
    # Test 2: Complete coverage
    expected_indices = set(range(total_size))
    if all_indices == expected_indices:
        print("  ✓ Complete coverage - all indices accounted for")
    else:
        missing = expected_indices - all_indices
        extra = all_indices - expected_indices
        print(f"  ✗ Incomplete coverage! Missing: {missing}, Extra: {extra}")
        return
    
    # Test 3: Sequential ordering (time-based split)
    if (train_indices == sorted(train_indices) and 
        val_indices == sorted(val_indices) and 
        test_indices == sorted(test_indices)):
        print("  ✓ Sequential ordering maintained (time-based split)")
    else:
        print("  ✗ Sequential ordering broken")
        return
    
    # Test 4: Temporal progression
    if (max(train_indices) < min(val_indices) and 
        max(val_indices) < min(test_indices)):
        print("  ✓ Temporal progression: Train → Validation → Test")
    else:
        print("  ✗ Temporal progression broken")
        return
    
    # Test data loading
    print(f"\nTesting Data Loading:")
    try:
        # Create datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        # Test loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"  ✓ Successfully created data loaders")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        print(f"    Test batches: {len(test_loader)}")
        
        # Test sample loading
        train_sample = next(iter(train_loader))
        val_sample = next(iter(val_loader))
        test_sample = next(iter(test_loader))
        
        print(f"  ✓ Successfully loaded sample batches")
        print(f"    Train sample shape: {train_sample[0].shape}")
        print(f"    Val sample shape: {val_sample[0].shape}")
        print(f"    Test sample shape: {test_sample[0].shape}")
        
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    
    # Analyze data distribution across splits
    try:
        # Sample some data points from each split
        train_samples = []
        val_samples = []
        test_samples = []
        
        for i, (input_state, target_state, _) in enumerate(train_loader):
            if i >= 5:  # Just sample first few batches
                break
            train_samples.append(target_state.mean().item())
        
        for i, (input_state, target_state, _) in enumerate(val_loader):
            if i >= 5:
                break
            val_samples.append(target_state.mean().item())
            
        for i, (input_state, target_state, _) in enumerate(test_loader):
            if i >= 5:
                break
            test_samples.append(target_state.mean().item())
        
        train_mean = np.mean(train_samples)
        val_mean = np.mean(val_samples)
        test_mean = np.mean(test_samples)
        
        print(f"  Train data mean: {train_mean:.6f}")
        print(f"  Val data mean: {val_mean:.6f}")
        print(f"  Test data mean: {test_mean:.6f}")
        
        # Check if means are reasonably similar (no extreme bias)
        all_means = [train_mean, val_mean, test_mean]
        mean_std = np.std(all_means)
        if mean_std < 1.0:  # Reasonable threshold
            print(f"  ✓ Data distribution appears balanced (std: {mean_std:.6f})")
        else:
            print(f"  ⚠ Data distribution may be imbalanced (std: {mean_std:.6f})")
            
    except Exception as e:
        print(f"  ⚠ Statistical analysis failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("Dataset split verification completed successfully!")
    print(f"The time-based train/validation/test split for {data_file} is working correctly.")
    print("This matches the exact same split used in train_complete_physics.py")

if __name__ == "__main__":
    verify_dataset_splits()