import pytest
import torch

from src.pokedec.model import get_model


def test_model():
    num_classes = 1000
    model = get_model(num_classes=num_classes)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (
        1,
        num_classes,
    ), f"Ecxpected output shape (1, {num_classes}), but got {y.shape}"


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_model_batch_size(batch_size):
    num_classes = 1000
    model = get_model(num_classes=num_classes)
    x = torch.randn(batch_size, 3, 128, 128)
    y = model(x)
    assert y.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape ({batch_size}, {num_classes}), but got {y.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
