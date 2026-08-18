import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from lungai.config import BATCH_SIZE, DISEASE_LABELS, MODEL_DIR, NUM_WORKERS
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import train_transform, validation_transform
from lungai.models.chest_xray_model import create_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(train_csv: str, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    positives = torch.tensor(
        [float(df[label].sum()) for label in DISEASE_LABELS],
        dtype=torch.float32,
    )
    negatives = float(len(df)) - positives
    weights = negatives / positives.clamp_min(1.0)
    return weights.to(device)


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
    device = get_device()
    print(f"device={device}")

    train_ds = ChestXrayDataset(train_csv, transform=train_transform)
    val_ds = ChestXrayDataset(val_csv, transform=validation_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = create_model(len(DISEASE_LABELS), pretrained=True).to(device)

    pos_weight = compute_pos_weight(train_csv, device)
    print(
        "pos_weight="
        + ", ".join(
            f"{label}:{weight:.2f}"
            for label, weight in zip(DISEASE_LABELS, pos_weight.detach().cpu().tolist())
        )
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
