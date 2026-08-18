from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from lungai.config import DISEASE_LABELS


class ChestXrayDataset(Dataset):
    """Dataset for CSV files with an image path and binary disease columns."""

    def __init__(self, csv_path: str | Path, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        self.transform = transform

        required = {"path", *DISEASE_LABELS}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        labels = torch.tensor(
            [float(row[label]) for label in DISEASE_LABELS], dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, labels
