import pandas as pd
from PIL import Image

from lungai.config import DISEASE_LABELS
from lungai.data.dataset import ChestXrayDataset
from lungai.data.transforms import validation_transform


def test_dataset_loading(tmp_path) -> None:
    image_path = tmp_path / "xray.png"
    Image.new("RGB", (32, 32), color="black").save(image_path)

    row = {"path": str(image_path)}
    row.update({label: index % 2 for index, label in enumerate(DISEASE_LABELS)})
    csv_path = tmp_path / "samples.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    dataset = ChestXrayDataset(csv_path, transform=validation_transform)
    image, labels = dataset[0]

    assert image.shape == (3, 224, 224)
    assert labels.shape == (len(DISEASE_LABELS),)
