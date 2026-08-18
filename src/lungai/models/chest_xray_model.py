import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a DenseNet121 multi-label classifier.

    Args:
        num_classes: Number of output disease labels.
        pretrained: Whether to initialize with ImageNet weights.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
