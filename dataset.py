# dataset.py

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
# Import transforms if you plan to use them (e.g., for data augmentation)
from torchvision import transforms 

class CIFAR10Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing the cifar10.pkl file.
    
    Loads image data from the specified split ('x_train', 'x_val', 'x_test') 
    and converts it into the PyTorch standard (C, H, W).
    """
    def __init__(self, pkl_path: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        Initializes the dataset by loading the data split from the .pkl file.
        
        Args:
            pkl_path (str): The full path to the cifar10.pkl file.
            split (str): Which data split to use ('train', 'val', or 'test').
            transform (transforms.Compose, optional): PyTorch data transformations.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid split. Must be 'train', 'val', or 'test'.")

        # 1. Load Data from Pickle
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except FileNotFoundError:
            raise FileNotFoundError(f"PKL file not found at: {pkl_path}")
        # 2. Select Data Split
        if split == 'train':
            self.images = data['x_train']
            self.labels = data['y_train'].ravel()
        elif split == 'val':
            self.images = data['x_val']
            self.labels = data['y_val'].ravel()
        elif split == 'test':
            self.images = data['x_test']
            self.labels = data['y_test'].ravel()

        self.transform = transform
        
        # 3. Transpose for PyTorch (H, W, C) -> (C, H, W)
        # The first dimension (0) is the number of samples (N), which remains in place.
        self.images = np.transpose(self.images, (0, 3, 1, 2))


    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.labels)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one sample of data (image and label) at the given index.
        """
        image_np = self.images[idx]
        label_np = self.labels[idx]
        # import pdb 
        # pdb.set_trace()

        # Convert to PyTorch tensors
        # Image is converted to FloatTensor (necessary for normalization/model input)
        # Label is converted to LongTensor (necessary for CrossEntropyLoss)
        image = torch.from_numpy(image_np).float()
        label = torch.tensor(label_np).long()
    
        # Apply the optional transform
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Utility Function for DataLoaders ---

def get_loaders(
    pkl_path: str, 
    batch_size: int, 
    num_workers: int = 1, 
    transforms_dict: Optional[Dict[str, transforms.Compose]] = None
) -> Dict[str, DataLoader]:
    """
    Creates and returns DataLoader instances for all splits.
    
    Args:
        pkl_path (str): Path to the cifar10.pkl file.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.
        transforms_dict (Dict[str, transforms.Compose], optional): A dictionary 
            containing transforms for 'train', 'val', and 'test' splits.
            
    Returns:
        Dict[str, DataLoader]: A dictionary with keys 'train', 'val', 'test'.
    """
    if transforms_dict is None:
        # Default transforms for normalization (CIFAR-10 means/stds) and conversion
        # NOTE: The custom Dataset handles (H, W, C) -> (C, H, W) and numpy -> tensor,
        # so these simple transforms are applied AFTER those steps.
        # We need to re-scale the 0-255 images to 0-1 range for normalization
        cifar10_mean = [0.4914, 0.4822, 0.4465]
        cifar10_std  = [0.2023, 0.1994, 0.2010]
        
        default_transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),  # Rescale to 0-1
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        transforms_dict = {
            'train': default_transform,
            'val': default_transform,
            'test': default_transform,
        }
        
    # 1. Create Datasets
    train_dataset = CIFAR10Dataset(pkl_path, 'train', transform=transforms_dict['train'])
    val_dataset   = CIFAR10Dataset(pkl_path, 'val',   transform=transforms_dict['val'])
    test_dataset  = CIFAR10Dataset(pkl_path, 'test',  transform=transforms_dict['test'])

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle training data
        # num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, # Do not shuffle validation data
        # num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, # Do not shuffle test data
        # num_workers=num_workers,
    )
    
    return {
        'train': train_loader, 
        'val': val_loader, 
        'test': test_loader
    }
