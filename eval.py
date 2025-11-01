import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from models.classification_model import ClassificationModel
from dataset.tiny_imagenet import TinyImageNetDataset
from dataset.augmentation import get_val_transforms
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for image classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': running_loss/total})

    return running_loss / total, 100. * correct / total

def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    val_dataset = TinyImageNetDataset(
        root_dir=args.data_dir,
        split='val',
        transform=get_val_transforms(config['input_size'])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create model
    model = ClassificationModel(
        backbone=config['backbone'],
        head=config['head'],
        out_features=config['out_features'],
        num_classes=config['num_classes']
    )
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint '{args.checkpoint}'")
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint '{args.checkpoint}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.checkpoint}'")
            return
    else:
        print("Please specify a checkpoint to evaluate using --checkpoint")
        return

    # Validate the model
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Print statistics
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

if __name__ == '__main__':
    main()