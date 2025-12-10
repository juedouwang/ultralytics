"""Minimal CNN example for handwritten digit classification on MNIST.

This script trains a small convolutional neural network using PyTorch and torchvision.
Run directly to download MNIST, train for a few epochs, and report accuracy:
    python examples/mnist_cnn.py --epochs 3 --batch-size 128
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistCNN(nn.Module):
    """Simple convolutional neural network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run a forward pass."""
        return self.classifier(self.features(x))


def get_loaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders for MNIST."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for a single epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(model: nn.Module, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    """Evaluate accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train_mnist(
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    data_root: Path | None = None,
    device: torch.device | None = None,
) -> None:
    """Train the CNN on MNIST and report final accuracy."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = data_root or Path.home() / ".cache" / "mnist"

    train_loader, test_loader = get_loaders(data_root, batch_size)
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, test_acc={acc:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a simple CNN on the MNIST handwritten digit dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--data-root", type=Path, default=None, help="Directory to store the MNIST dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_mnist(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, data_root=args.data_root)
