import importlib.util
from pathlib import Path

import torch

MNIST_CNN_PATH = Path(__file__).resolve().parents[1] / "examples" / "mnist_cnn.py"
spec = importlib.util.spec_from_file_location("mnist_cnn", MNIST_CNN_PATH)
mnist_cnn = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mnist_cnn)


def test_mnist_cnn_forward_shape():
    model = mnist_cnn.MnistCNN()
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    assert out.shape == (4, 10)


def test_evaluate_returns_probability_range():
    model = mnist_cnn.MnistCNN()
    images = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([1, 2])
    accuracy = mnist_cnn.evaluate(model, [(images, labels)], device=torch.device("cpu"))
    assert 0.0 <= accuracy <= 1.0
