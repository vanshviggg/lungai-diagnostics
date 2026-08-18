import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from lungai.config import BATCH_SIZE, DISEASE_LABELS, MODEL_DIR
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import train_transform, validation_transform
from lungai.models.chest_xray_model import create_model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def train(train_csv: str, val_csv: str, epochs: int = 5, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ChestXrayDataset(train_csv, transform=train_transform)
    val_ds = ChestXrayDataset(val_csv, transform=validation_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(len(DISEASE_LABELS), pretrained=True).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / "best_model.pt"
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, device)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"saved={best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.train_csv, args.val_csv, epochs=args.epochs, lr=args.lr)
