"""
Kiểm thử đơn vị cho các kiến trúc mô hình (MLP & CNN).

Các bài test kiểm chứng:
1. Shape của đầu ra (Output tensor shape) với các batch size khác nhau.
2. Tính hữu hạn của giá trị đầu ra (Finite values: không chứa NaN hoặc Inf).
3. Smoke test lan truyền xuôi và ngược (Forward & Backward pass) trên dữ liệu giả lập (synthetic data).
4. Khả năng cập nhật trọng số của optimizer sau một bước gradient descent.

Tất cả các bài test chạy hoàn toàn trên CPU và KHÔNG tải dataset thực tế về máy.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lab.fashion_mnist.model import FashionMNISTMLP
from pytorch_lab.cifar10.model import CIFAR10CNN


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_fashion_mnist_mlp_shape(batch_size: int) -> None:
    """Kiểm tra MLP nhận input (B, 1, 28, 28) và trả về (B, 10)."""
    model = FashionMNISTMLP(num_classes=10)
    model.eval()

    # Dữ liệu ảnh giả lập dạng 4D [B, 1, 28, 28]
    dummy_input_4d = torch.randn(batch_size, 1, 28, 28)
    output_4d = model(dummy_input_4d)
    assert output_4d.shape == (batch_size, 10), (
        f"Kỳ vọng shape ({batch_size}, 10) nhưng nhận ({output_4d.shape})"
    )

    # Dữ liệu ảnh giả lập đã trải phẳng dạng 2D [B, 784]
    dummy_input_2d = torch.randn(batch_size, 784)
    output_2d = model(dummy_input_2d)
    assert output_2d.shape == (batch_size, 10)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_cifar10_cnn_shape(batch_size: int) -> None:
    """Kiểm tra CNN nhận input (B, 3, 32, 32) và trả về (B, 10)."""
    model = CIFAR10CNN(num_classes=10)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (batch_size, 10), (
        f"Kỳ vọng shape ({batch_size}, 10) nhưng nhận ({output.shape})"
    )


def test_models_output_finite() -> None:
    """Kiểm tra giá trị đầu ra của cả 2 mô hình là hữu hạn (không bị NaN hoặc Inf)."""
    mlp = FashionMNISTMLP()
    cnn = CIFAR10CNN()

    mlp_out = mlp(torch.randn(4, 1, 28, 28))
    cnn_out = cnn(torch.randn(4, 3, 32, 32))

    assert torch.all(torch.isfinite(mlp_out)), "MLP output chứa giá trị không hợp lệ (NaN hoặc Inf)!"
    assert torch.all(torch.isfinite(cnn_out)), "CNN output chứa giá trị không hợp lệ (NaN hoặc Inf)!"


def test_mlp_forward_backward_smoke() -> None:
    """Smoke test vòng lặp forward + backward + step cho MLP trên batch giả lập."""
    model = FashionMNISTMLP()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dữ liệu giả lập
    inputs = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,))

    # Lưu trọng số trước khi cập nhật
    initial_weight = model.fc1.weight.clone()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Kiểm tra gradient đã được tính
    assert model.fc1.weight.grad is not None
    assert torch.all(torch.isfinite(model.fc1.weight.grad))

    optimizer.step()

    # Kiểm tra trọng số đã thay đổi sau bước tối ưu
    assert not torch.equal(initial_weight, model.fc1.weight), "Trọng số không được cập nhật sau optimizer.step()!"


def test_cnn_forward_backward_smoke() -> None:
    """Smoke test vòng lặp forward + backward + step cho CNN trên batch giả lập."""
    model = CIFAR10CNN()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))

    initial_weight = model.conv1.weight.clone()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    assert model.conv1.weight.grad is not None
    assert torch.all(torch.isfinite(model.conv1.weight.grad))

    optimizer.step()
    assert not torch.equal(initial_weight, model.conv1.weight), "Trọng số CNN không thay đổi sau optimizer.step()!"
