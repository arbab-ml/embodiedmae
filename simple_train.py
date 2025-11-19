"""
Simple training script for EmbodiedMAE with synthetic data.

Minimal delta from original embodiedmae codebase.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from embodied_mae import EmbodiedMAEForMaskedImageModeling
from scripts.simple_synthetic_dataset import SimpleSyntheticDataset


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_rgb_loss = 0.0
    total_depth_loss = 0.0
    total_pc_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        pc = batch['pc'].to(device)

        # Forward pass
        outputs = model(rgb, depth, pc, add_mask=True)

        # Compute loss
        rgb_loss, depth_loss, pc_loss = model.decoder.get_loss(
            outputs.decoder_output, rgb, depth, pc
        )
        loss = rgb_loss + depth_loss + pc_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_depth_loss += depth_loss.item()
        total_pc_loss += pc_loss.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rgb': f"{rgb_loss.item():.4f}",
            'depth': f"{depth_loss.item():.4f}",
            'pc': f"{pc_loss.item():.4f}",
        })

    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    avg_pc_loss = total_pc_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'rgb_loss': avg_rgb_loss,
        'depth_loss': avg_depth_loss,
        'pc_loss': avg_pc_loss,
    }


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_rgb_loss = 0.0
    total_depth_loss = 0.0
    total_pc_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Validation"):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        pc = batch['pc'].to(device)

        outputs = model(rgb, depth, pc, add_mask=True)

        rgb_loss, depth_loss, pc_loss = model.decoder.get_loss(
            outputs.decoder_output, rgb, depth, pc
        )
        loss = rgb_loss + depth_loss + pc_loss

        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_depth_loss += depth_loss.item()
        total_pc_loss += pc_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    avg_pc_loss = total_pc_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'rgb_loss': avg_rgb_loss,
        'depth_loss': avg_depth_loss,
        'pc_loss': avg_pc_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple EmbodiedMAE Training")
    parser.add_argument('--model', type=str, default='base', choices=['base', 'large', 'giant'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='./outputs/simple_train')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading EmbodiedMAE-{args.model}...")
    model_name = f"ZibinDong/embodiedmae-{args.model}"
    model = EmbodiedMAEForMaskedImageModeling.from_pretrained(model_name)
    model = model.to(device)
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create datasets
    print(f"\nCreating synthetic datasets ({args.num_samples} samples)...")
    train_dataset = SimpleSyntheticDataset(
        num_samples=int(args.num_samples * 0.9),
        seed=42
    )
    val_dataset = SimpleSyntheticDataset(
        num_samples=int(args.num_samples * 0.1),
        seed=123
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Setup scheduler (cosine with warmup)
    warmup_epochs = int(args.epochs * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device, epoch)

        # Update scheduler
        scheduler.step()

        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(RGB: {train_metrics['rgb_loss']:.4f}, "
              f"Depth: {train_metrics['depth_loss']:.4f}, "
              f"PC: {train_metrics['pc_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} "
              f"(RGB: {val_metrics['rgb_loss']:.4f}, "
              f"Depth: {val_metrics['depth_loss']:.4f}, "
              f"PC: {val_metrics['pc_loss']:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }, checkpoint_path)
            print(f"  ✅ Saved best model (loss: {best_val_loss:.4f})")

        # Save latest
        latest_path = output_dir / 'latest_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
        }, latest_path)

    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
