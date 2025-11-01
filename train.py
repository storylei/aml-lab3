import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.classification_model import ClassificationModel
from dataset.tiny_imagenet import TinyImageNetDataset
from dataset.augmentation import get_train_transforms, get_val_transforms
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200', help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint to disk"""
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        
        # Use torch.amp.autocast to automatically cast operations to appropriate data types (FP16 or BF16)
        # to improve performance (increase speed and reduce vram usage ) while maintaining accuracy. The device_type is set to 'cuda' to indicate that
        # this is for GPU usage. The 'enabled' flag is set according to the config file, determining whether
        # mixed precision is used or not.
        with torch.amp.autocast(device_type=str(device), enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total})
    
    return running_loss / total, 100. * correct / total

def validate(model, dataloader, criterion, device, use_amp):
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
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
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

    # Initialize wandb after loading config
    wandb.init(project="image-classification", config={
        "learning_rate": config['learning_rate'],
        "batch_size": config['batch_size'],
        "epochs": config['epochs'],
        "model": "ClassificationModel"
    })

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(device,enabled=config['use_amp'])

    # Create datasets and dataloaders
    train_dataset = TinyImageNetDataset(
        root_dir=args.data_dir,
        split='train',
        transform=get_train_transforms(config['input_size'])
    )

    val_dataset = TinyImageNetDataset(
        root_dir=args.data_dir,
        split='val',
        transform=get_val_transforms(config['input_size'])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch: {epoch+1}/{config['epochs']}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, config['use_amp'])

        # Log training metrics to wandb
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch+1})

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, config['use_amp'])

        # Log validation metrics to wandb
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch+1})

        # Update learning rate
        scheduler.step()

        # Print statistics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, checkpoint_dir)

        # Save model to wandb
        if is_best:
            wandb.save(os.path.join(checkpoint_dir, 'model_best.pth'))

    print(f"Training completed. Best accuracy: {best_acc:.2f}%")

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()