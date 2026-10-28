"""
Package phân loại ảnh CIFAR-10 sử dụng mạng tích chập Convolutional Neural Network (CNN).
"""

from pytorch_lab.cifar10.model import CIFAR10CNN
from pytorch_lab.cifar10.data import get_cifar10_dataloaders, CIFAR10_CLASSES

__all__ = ["CIFAR10CNN", "get_cifar10_dataloaders", "CIFAR10_CLASSES"]
