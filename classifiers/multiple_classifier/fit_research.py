"""Controlled EfficientNet ablations for the SCC imbalance research experiment.

All variants use the same ImageFolder classes, weighted sampler, optimizer,
schedule, seed, train/held-out split, and checkpoint-selection rule. The focal
variant changes only the loss relative to attention_ce.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from cnn import CustomClassifier
from cnn_attention import AttentionClassifier
from utils import create_transforms, get_device


EXPECTED_CLASSES = ["BCC", "MEL", "SCC"]
EXPERIMENTS = ("plain_ce", "attention_ce", "attention_focal")


class FocalLoss(nn.Module):
    """Multiclass focal loss; balanced sampling handles class-frequency bias."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probabilities = torch.log_softmax(logits, dim=1)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        return (((1.0 - pt) ** self.gamma) * (-log_pt)).mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_loaders(train_dir, val_dir, batch_size, seed):
    transform = create_transforms()
    train_set = ImageFolder(root=train_dir, transform=transform)
    val_set = ImageFolder(root=val_dir, transform=transform)
    if train_set.classes != EXPECTED_CLASSES or val_set.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected class order {EXPECTED_CLASSES}; got train={train_set.classes}, val={val_set.classes}"
        )
    counts = torch.bincount(torch.tensor(train_set.targets), minlength=len(EXPECTED_CLASSES))
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[torch.tensor(train_set.targets)]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_set, val_set, train_loader, val_loader, counts.tolist()


def build_model(experiment, device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    input_size = model.classifier[1].in_features
    if experiment == "plain_ce":
        model.classifier[1] = CustomClassifier(
            input_size=input_size, hidden_size1=1024, hidden_size2=512, num_classes=3
        )
    else:
        model.classifier = AttentionClassifier(
            input_size=input_size, hidden_size1=1024, hidden_size2=512, num_classes=3
        )
    return model.to(device)


def evaluate(model, loader, device, save_predictions=None):
    model.eval()
    matrix = [[0 for _ in EXPECTED_CLASSES] for _ in EXPECTED_CLASSES]
    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for truth, prediction in zip(labels.tolist(), predictions.tolist()):
                matrix[truth][prediction] += 1
            all_labels.extend(labels.tolist())
            all_probabilities.extend(probabilities.tolist())

    supports = [sum(row) for row in matrix]
    predicted_counts = [sum(matrix[r][c] for r in range(3)) for c in range(3)]
    precision, recall, f1 = [], [], []
    for index in range(3):
        p = matrix[index][index] / predicted_counts[index] if predicted_counts[index] else 0.0
        r = matrix[index][index] / supports[index] if supports[index] else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    total = sum(supports)
    accuracy = sum(matrix[i][i] for i in range(3)) / total if total else 0.0
    expected = sum(supports[i] * predicted_counts[i] for i in range(3)) / (total * total) if total else 0.0
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "scc_precision": precision[2],
        "scc_recall": recall[2],
        "scc_f1": f1[2],
        "cohen_kappa": kappa,
        "confusion_matrix": matrix,
        "classes": EXPECTED_CLASSES,
        "support": supports,
    }
    if save_predictions:
        with Path(save_predictions).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "actual", "predicted", "p_BCC", "p_MEL", "p_SCC"])
            for (image_path, _label), truth, probabilities in zip(
                loader.dataset.samples, all_labels, all_probabilities
            ):
                predicted = int(np.argmax(probabilities))
                writer.writerow([
                    Path(image_path).name,
                    EXPECTED_CLASSES[truth],
                    EXPECTED_CLASSES[predicted],
                    *[f"{p:.8f}" for p in probabilities],
                ])
    return metrics


def train_one(experiment, args, device):
    set_seed(args.seed)
    train_set, val_set, train_loader, val_loader, counts = make_loaders(
        args.train_dir, args.val_dir, args.batch_size, args.seed
    )
    model = build_model(experiment, device)
    criterion = FocalLoss(gamma=args.gamma) if experiment == "attention_focal" else nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = optim.Adam(model.parameters(), lr=args.max_lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        step_size_up=4 * len(train_loader),
        mode="triangular2",
        cycle_momentum=False,
    )

    run_dir = Path(args.save_dir) / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pth"
    best_macro_f1 = -1.0
    best_epoch = 0
    history = []
    stale_epochs = 0

    print(f"\n=== {experiment} | seed={args.seed} | device={device} ===")
    print(f"Classes: {EXPECTED_CLASSES}; training counts: {counts}; validation images: {len(val_set)}")
    print("Sampler: inverse-frequency weighted random sampler; validation: natural class distribution")

    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        running_loss = 0.0
        examples_seen = 0
        for images, labels in tqdm(train_loader, desc=f"{experiment} epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * images.size(0)
            examples_seen += images.size(0)

        metrics = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(examples_seen, 1),
            "validation": metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch + 1} train_loss={epoch_record['train_loss']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} scc_recall={metrics['scc_recall']:.4f} "
            f"acc={metrics['accuracy']:.4f} elapsed={time.time() - started:.1f}s"
        )

        # Use the held-out tuning split, never training loss, to choose a checkpoint.
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1

        if args.patience and stale_epochs >= args.patience:
            print(f"Early stop after {args.patience} epochs without validation macro-F1 improvement.")
            break

    best_model = build_model(experiment, device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    best_metrics = evaluate(
        best_model,
        val_loader,
        device,
        save_predictions=run_dir / "validation_predictions.csv",
    )
    manifest = {
        "experiment": experiment,
        "model": "EfficientNet-B0",
        "head": "plain_custom_classifier" if experiment == "plain_ce" else "feature_attention",
        "loss": "cross_entropy" if experiment != "attention_focal" else "multiclass_focal",
        "focal_gamma": args.gamma if experiment == "attention_focal" else None,
        "sampler": "inverse_frequency_weighted_random_sampler",
        "seed": args.seed,
        "mixed_precision": use_amp,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "checkpoint_selection": "highest held-out validation macro-F1",
        "train_dir": str(Path(args.train_dir)),
        "validation_dir": str(Path(args.val_dir)),
        "train_images": len(train_set),
        "validation_images": len(val_set),
        "training_class_counts": dict(zip(EXPECTED_CLASSES, counts)),
        "validation_class_counts": dict(zip(EXPECTED_CLASSES, best_metrics["support"])),
        "validation_metrics": best_metrics,
        "epoch_history": history,
        "note": "Internal tuning results; do not report as final test performance.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved run manifest and validation predictions: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run controlled SCC imbalance ablations.")
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    parser.add_argument("--train_dir", default="data_mult/train/")
    parser.add_argument("--val_dir", default="data_mult/val/")
    parser.add_argument("--save_dir", default="out/research_ablation/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use mixed precision on CUDA (disable with --no-amp)")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.gamma < 0:
        parser.error("epochs and batch_size must be positive; gamma cannot be negative")
    device = get_device()
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for experiment in experiments:
        train_one(experiment, args, device)


if __name__ == "__main__":
    main()
