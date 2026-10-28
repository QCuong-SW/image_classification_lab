"""
Package phân loại ảnh FashionMNIST sử dụng mạng MLP (Multi-Layer Perceptron).
"""

from pytorch_lab.fashion_mnist.model import FashionMNISTMLP
from pytorch_lab.fashion_mnist.data import get_fashion_mnist_dataloaders, FASHION_MNIST_CLASSES

__all__ = ["FashionMNISTMLP", "get_fashion_mnist_dataloaders", "FASHION_MNIST_CLASSES"]
