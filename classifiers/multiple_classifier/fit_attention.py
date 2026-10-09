import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from sklearn.metrics import cohen_kappa_score

from cnn_attention import AttentionClassifier
from utils import get_device, create_transforms


def load_data(path, transform, batch_size):
    dataset = ImageFolder(
        root=path,
        transform=transform
    )

    # Count samples in each class
    class_counts = torch.bincount(
        torch.tensor(dataset.targets)
    )

    print("Loaded dataset:")
    print(f" - Number of images: {len(dataset)}")
    print(f" - Number of classes: {len(dataset.classes)}")
    print(f" - Class names: {dataset.classes}")

    for idx, class_name in enumerate(dataset.classes):
        print(
            f" -- {class_name}: "
            f"{class_counts[idx].item()} images"
        )

    print(f" - Batch size: {batch_size}")

    # Inverse-frequency class weights
    class_weights = 1.0 / class_counts.float()

    # Weight for every individual sample
    sample_weights = torch.tensor(
        [
            class_weights[label]
            for label in dataset.targets
        ],
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    print(f"Class weights: {class_weights}")

    return loader


def initialize_attention_model(device):
    print("Initializing EfficientNet-B0 with Attention...")

    # Pretrained EfficientNet-B0
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    # Get feature dimension
    input_size = model.classifier[1].in_features

    # Replace original classifier
    model.classifier = AttentionClassifier(
        input_size=input_size,
        hidden_size1=1024,
        hidden_size2=512,
        num_classes=3
    )

    model = model.to(device)

    print(
        f"EfficientNet-B0 feature size: {input_size}"
    )

    print(
        "Attention classifier attached successfully."
    )

    return model


def train_model(
    model,
    train_loader,
    device,
    number_epochs,
    save_dir
):

    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    steps_per_epoch = len(train_loader)

    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=1e-4,
        step_size_up=4 * steps_per_epoch,
        mode="triangular2",
        cycle_momentum=False
    )

    best_loss = float("inf")

    for epoch in range(number_epochs):

        start_time = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        print(
            f"\nTraining EfficientNet-B0 + Attention "
            f"- Epoch {epoch + 1}/{number_epochs}"
        )

        print(
            f"Current Learning Rate: "
            f"{optimizer.param_groups[0]['lr']}"
        )

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            unit="batch"
        )

        for inputs, labels in progress_bar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            scheduler.step()

            running_loss += (
                loss.item() * inputs.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

            all_predictions.extend(
                predicted.detach().cpu().numpy()
            )

            accuracy = correct / total

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{accuracy:.4f}"
            )

        epoch_loss = (
            running_loss /
            len(train_loader.dataset)
        )

        epoch_accuracy = correct / total

        epoch_kappa = cohen_kappa_score(
            all_labels,
            all_predictions
        )

        epoch_time = time.time() - start_time

        print(
            f"\nEfficientNet-B0 + Attention "
            f"Epoch {epoch + 1} "
            f"- Loss: {epoch_loss:.4f}, "
            f"Acc: {epoch_accuracy:.4f}, "
            f"Kappa: {epoch_kappa:.4f}"
        )

        # Save best training-loss checkpoint
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            save_path = os.path.join(
                save_dir,
                "EfficientNet_Attention_best.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"Saved improved model at: "
                f"{save_path}"
            )

        print(
            f"Epoch {epoch + 1} Completed - "
            f"Time: {epoch_time:.2f}s"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 with Attention"
    )

    parser.add_argument(
        "--number_epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default="data_mult/train/"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/proposed_attention/"
    )

    args = parser.parse_args()

    device = get_device()

    print(
        f"Using device: {device}"
    )

    transform = create_transforms()

    train_loader = load_data(
        args.train_dir,
        transform,
        args.batch_size
    )

    model = initialize_attention_model(
        device
    )

    train_model(
        model,
        train_loader,
        device,
        args.number_epochs,
        args.save_dir
    )


if __name__ == "__main__":
    main()