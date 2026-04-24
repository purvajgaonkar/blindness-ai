"""
train.py
========
Full training script for EfficientNet-B4 DR severity classifier.
Supports: mixed-precision training, WeightedRandomSampler, QWK metric,
          CosineAnnealingLR, label smoothing loss, and comprehensive output artifacts.

Usage:
    python train.py --data_path /path/to/aptos2019_data
    python train.py --data_path /path/to/aptos2019_data --epochs 30 --batch_size 16 --lr 1e-4
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model   import DRClassifier, LabelSmoothingCrossEntropy, CLASS_NAMES
from dataset import build_dataloaders
from utils   import (
    quadratic_weighted_kappa, plot_confusion_matrix,
    plot_training_curves, plot_roc_curves,
    save_classification_report, save_history, logger
)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B4 for Diabetic Retinopathy Detection"
    )
    parser.add_argument('--data_path',  type=str,   required=True,
                        help='Root directory of APTOS dataset (contains train.csv, valid.csv, etc.)')
    parser.add_argument('--epochs',     type=int,   default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int,   default=16,
                        help='Batch size per device (default: 16)')
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--img_size',   type=int,   default=380,
                        help='Input image size in pixels (default: 380 for EfficientNet-B4)')
    parser.add_argument('--num_workers',type=int,   default=4,
                        help='DataLoader worker threads (default: 4)')
    parser.add_argument('--output_dir', type=str,   default='outputs',
                        help='Directory for saved artifacts (default: outputs/)')
    parser.add_argument('--model_dir',  type=str,   default='models',
                        help='Directory for saved model checkpoints (default: models/)')
    parser.add_argument('--resume',     type=str,   default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='AdamW weight decay (default: 1e-5)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    return parser.parse_args()


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, epoch, num_epochs
):
    """Run one full training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]",
                leave=False, ncols=100)

    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == 'cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         avg_loss=f"{total_loss/num_batches:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Run validation epoch.

    Returns
    -------
    val_loss   : float
    accuracy   : float
    qwk        : float
    all_labels : np.ndarray
    all_preds  : np.ndarray
    all_probs  : np.ndarray  (N, 5)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds  = []
    all_probs  = []

    pbar = tqdm(loader, desc="Validation", leave=False, ncols=100)

    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == 'cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)

    accuracy   = float(np.mean(all_labels == all_preds))
    qwk        = quadratic_weighted_kappa(all_labels, all_preds)
    avg_loss   = total_loss / max(len(loader), 1)

    return avg_loss, accuracy, qwk, all_labels, all_preds, all_probs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir,  exist_ok=True)
    best_model_path = os.path.join(args.model_dir, 'efficientnet_b4_dr_best.pth')

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, class_weights = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Initialising EfficientNet-B4 model...")
    model = DRClassifier(num_classes=5, pretrained=True)
    model.to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing, num_classes=5
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch   = 0
    best_qwk      = -1.0
    history = {
        'train_loss':   [],
        'val_loss':     [],
        'val_accuracy': [],
        'val_qwk':      [],
        'lr':           [],
    }

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_qwk    = ckpt.get('best_qwk', -1.0)
        if 'history' in ckpt:
            history = ckpt['history']
        logger.info(f"Resumed at epoch {start_epoch}, best QWK = {best_qwk:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Starting training for {args.epochs} epochs")
    logger.info(f"  LR={args.lr}, BS={args.batch_size}, ImgSize={args.img_size}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, args.epochs
        )

        # Validate
        val_loss, accuracy, qwk, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch results
        epoch_time = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"Acc={accuracy:.4f} | QWK={qwk:.4f} | "
            f"LR={current_lr:.2e} | Time={epoch_time:.1f}s"
        )

        # Update history
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_accuracy'].append(round(accuracy, 4))
        history['val_qwk'].append(round(qwk, 4))
        history['lr'].append(current_lr)

        # Save best model based on QWK
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_qwk':            best_qwk,
                'val_loss':            val_loss,
                'val_accuracy':        accuracy,
                'history':             history,
                'args': vars(args),
            }, best_model_path)
            logger.info(f"  ✓ New best model saved! QWK={best_qwk:.4f}")

        # Save history every epoch (for interruption recovery)
        history_path = os.path.join(args.output_dir, 'training_history.json')
        save_history(history, history_path)

    # ── Post-training artifacts ────────────────────────────────────────────
    logger.info("\nGenerating evaluation artifacts...")

    # Reload best model for final evaluation
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    val_loss, accuracy, qwk, val_labels, val_preds, val_probs = validate(
        model, val_loader, criterion, device
    )

    logger.info(f"Final Validation — Accuracy: {accuracy:.4f} | QWK: {qwk:.4f}")

    # Confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )

    # ROC curves
    plot_roc_curves(
        val_labels, val_probs,
        save_path=os.path.join(args.output_dir, 'roc_curves.png')
    )

    # Classification report
    save_classification_report(
        val_labels, val_preds,
        save_path=os.path.join(args.output_dir, 'classification_report.txt'),
        extra_metrics={
            'Overall Accuracy':      f"{accuracy:.4f}",
            'Quadratic Weighted Kappa (QWK)': f"{qwk:.4f}",
            'Best Epoch QWK':        f"{best_qwk:.4f}",
        }
    )

    logger.info("\n" + "=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  Best model  : {best_model_path}")
    logger.info(f"  Final QWK   : {best_qwk:.4f}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
