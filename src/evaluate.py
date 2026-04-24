"""
evaluate.py
===========
Standalone evaluation script for the trained EfficientNet-B4 DR classifier.
Computes all metrics on a given split (val or test) and generates visual reports.

Usage:
    python evaluate.py --model_path models/efficientnet_b4_dr_best.pth
                       --data_path /path/to/aptos2019_data
                       --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model   import DRClassifier, CLASS_NAMES
from dataset import APTOSDataset, get_val_transforms
from utils   import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    save_classification_report, logger, quadratic_weighted_kappa
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DR classifier")
    parser.add_argument('--model_path', type=str,
                        default='models/efficientnet_b4_dr_best.pth')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--split',      type=str, default='val',
                        choices=['val', 'test'],
                        help="Dataset split to evaluate on (val or test)")
    parser.add_argument('--img_size',   type=int, default=380)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device):
    """
    Forward pass over entire dataset split.

    Returns
    -------
    labels : np.ndarray  (N,)   Ground-truth class indices
    preds  : np.ndarray  (N,)   Predicted class indices
    probs  : np.ndarray  (N, 5) Predicted probabilities
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    for images, labels, _ in tqdm(loader, desc="Evaluating", ncols=100):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from: {args.model_path}")
    model = DRClassifier(num_classes=5, pretrained=False)

    if os.path.isfile(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        logger.info("Model weights loaded successfully.")
    else:
        logger.warning(f"Checkpoint not found at {args.model_path}. Using random weights.")

    model.to(device)
    model.eval()

    # ── Build dataset ─────────────────────────────────────────────────────
    if args.split == 'val':
        csv_path = os.path.join(args.data_path, 'valid.csv')
        img_dir  = os.path.join(args.data_path, 'val_images')
    else:
        csv_path = os.path.join(args.data_path, 'test.csv')
        img_dir  = os.path.join(args.data_path, 'test_images')

    dataset = APTOSDataset(
        csv_path=csv_path, image_dir=img_dir,
        transform=get_val_transforms(args.img_size),
        preprocess=True, img_size=args.img_size
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Evaluating on {len(dataset)} samples ({args.split} split)")

    # ── Evaluate ──────────────────────────────────────────────────────────
    labels, preds, probs = run_evaluation(model, loader, device)

    # Filter out -1 labels (test set without annotations)
    valid_mask = labels >= 0
    labels = labels[valid_mask]
    preds  = preds[valid_mask]
    probs  = probs[valid_mask]

    if len(labels) == 0:
        logger.warning("No labelled samples found. Cannot compute metrics.")
        return

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs)
    logger.info(f"Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"QWK      : {metrics['qwk']:.4f}")
    logger.info("Per-class accuracy:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, metrics['per_class_accuracy'])):
        logger.info(f"  {name:20s}: {acc:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')
    )
    plot_roc_curves(
        labels, probs,
        save_path=os.path.join(args.output_dir, f'roc_curves_{args.split}.png')
    )
    save_classification_report(
        labels, preds,
        save_path=os.path.join(args.output_dir, f'classification_report_{args.split}.txt'),
        extra_metrics={
            'Overall Accuracy': f"{metrics['accuracy']:.4f}",
            'QWK':              f"{metrics['qwk']:.4f}",
        }
    )
    logger.info(f"Artifacts saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
