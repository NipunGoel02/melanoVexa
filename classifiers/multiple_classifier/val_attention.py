import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

from cnn_attention import AttentionClassifier
from utils import create_transforms, get_device


def initialize_attention_model(device):
    print("Initializing EfficientNet-B0 + Attention...")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    input_size = model.classifier[1].in_features
    print(f"EfficientNet-B0 feature size: {input_size}")

    model.classifier = AttentionClassifier(
        input_size=input_size,
        hidden_size1=1024,
        hidden_size2=512,
        num_classes=3
    )

    model = model.to(device)

    return model


def load_validation_data(val_dir, batch_size):
    transform = create_transforms()

    dataset = datasets.ImageFolder(
        root=val_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("\nValidation Dataset:")
    print(f"Number of images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    for class_name, class_index in dataset.class_to_idx.items():
        count = sum(
            1 for _, label in dataset.samples
            if label == class_index
        )
        print(f" - {class_name}: {count} images")

    return dataset, dataloader


def evaluate_model(model, dataloader, device, class_names):
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    kappa = cohen_kappa_score(
        all_labels,
        all_predictions
    )

    avg_loss = total_loss / len(all_labels)

    print("\n" + "=" * 60)
    print("EfficientNet-B0 + Attention Validation Results")
    print("=" * 60)

    print(f"Validation Loss : {avg_loss:.4f}")
    print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Cohen Kappa     : {kappa:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("Confusion Matrix:")
    print(cm)

    # SCC-specific result
    if "SCC" in class_names:

        scc_index = class_names.index("SCC")

        scc_tp = cm[scc_index, scc_index]

        scc_actual = cm[scc_index, :].sum()

        scc_predicted = cm[:, scc_index].sum()

        scc_recall = (
            scc_tp / scc_actual
            if scc_actual > 0
            else 0
        )

        scc_precision = (
            scc_tp / scc_predicted
            if scc_predicted > 0
            else 0
        )

        if scc_precision + scc_recall > 0:
            scc_f1 = (
                2 * scc_precision * scc_recall
                / (scc_precision + scc_recall)
            )
        else:
            scc_f1 = 0

        print("\nSCC Specific Results:")
        print(f"SCC Precision : {scc_precision:.4f}")
        print(f"SCC Recall    : {scc_recall:.4f}")
        print(f"SCC F1-score  : {scc_f1:.4f}")

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val_dir",
        type=str,
        default="data_mult/val/"
    )

    parser.add_argument(
        "--root_out",
        type=str,
        default="out/proposed_attention/"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    args = parser.parse_args()

    device = get_device()

    print(f"Using device: {device}")

    # Load validation data
    dataset, dataloader = load_validation_data(
        args.val_dir,
        args.batch_size
    )

    # Initialize model
    model = initialize_attention_model(device)

    # Checkpoint path
    checkpoint_path = os.path.join(
        args.root_out,
        "EfficientNet_Attention_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"\nLoading checkpoint:\n{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Handle both possible checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    print("Checkpoint loaded successfully.")

    # Evaluate
    evaluate_model(
        model,
        dataloader,
        device,
        dataset.classes
    )


if __name__ == "__main__":
    main()