from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from lungai.config import DISEASE_LABELS
from lungai.data.transforms import validation_transform
from lungai.models.chest_xray_model import create_model


class LungAIPredictor:
    """Load a trained model checkpoint and run chest X-ray inference."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = create_model(len(DISEASE_LABELS), pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> Dict[str, float]:
        image = Image.open(image_path).convert("RGB")
        tensor = validation_transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        return {
            label: float(probability)
            for label, probability in zip(DISEASE_LABELS, probabilities)
        }
