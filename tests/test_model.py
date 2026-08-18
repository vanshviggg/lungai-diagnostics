import torch

from lungai.models.chest_xray_model import create_model


def test_model_output_shape() -> None:
    model = create_model(num_classes=3, pretrained=False)
    model.eval()

    sample = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        output = model(sample)

    assert output.shape == (2, 3)
