"""
Module định nghĩa kiến trúc mạng nơ-ron tích chập (Convolutional Neural Network - CNN)
cho bài toán phân loại ảnh CIFAR-10.

KIẾN TRÚC MẠNG CNN BASELINE:
1. Conv2d Tầng 1:
   - Đầu vào: [B, 3, 32, 32] (3 kênh màu RGB).
   - Kernel 3x3, padding=1, stride=1 -> Giữ nguyên kích thước không gian: [B, conv1_channels, 32, 32].
   - ReLU Activation: Kích hoạt phi tuyến tính.
   - MaxPool2d(kernel_size=2, stride=2) -> Giảm độ phân giải không gian đi một nửa: [B, conv1_channels, 16, 16].
2. Conv2d Tầng 2:
   - Đầu vào: [B, conv1_channels, 16, 16].
   - Kernel 3x3, padding=1, stride=1 -> [B, conv2_channels, 16, 16].
   - ReLU Activation.
   - MaxPool2d(kernel_size=2, stride=2) -> [B, conv2_channels, 8, 8].
3. Flatten Layer:
   - Trải phẳng đặc trưng không gian: [B, conv2_channels * 8 * 8].
   - Với mặc định conv2_channels=64: 64 * 8 * 8 = 4096 đặc trưng.
4. Fully Connected (Linear) Tầng 1:
   - 4096 -> fc_hidden_dim (mặc định: 128).
   - ReLU Activation.
5. Fully Connected Tầng 2 (Output Logits):
   - fc_hidden_dim -> 10 (10 lớp của CIFAR-10, raw logits thô).

TẠI SAO PHIÊN BẢN BASELINE KHÔNG DÙNG BATCHNORM VÀ DROPOUT?
- Mục tiêu của dự án là học từ gốc rễ (first principles). Một kiến trúc thuần túy không có
  BatchNorm và Dropout giúp sinh viên thấy rõ dòng chảy của dữ liệu, hình dạng tensor sau từng bước
  pooling, và hiểu chính xác backpropagation cơ bản trước khi thêm các kỹ thuật regularization nâng cao.
- Trong phần thực nghiệm nâng cao (experiments), ta có thể so sánh hiệu quả khi thêm data augmentation
  hoặc thay đổi số kênh convolution.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Mạng nơ-ron tích chập (CNN) phân loại ảnh màu CIFAR-10.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        """
        Khởi tạo kiến trúc CNN.

        Args:
            in_channels: Số kênh ảnh đầu vào (3 cho ảnh màu RGB).
            conv1_channels: Số kênh bộ lọc tích chập tầng 1 (mặc định: 32).
            conv2_channels: Số kênh bộ lọc tích chập tầng 2 (mặc định: 64).
            fc_hidden_dim: Số nơ-ron tầng fully-connected ẩn (mặc định: 128).
            num_classes: Số lượng lớp phân loại đầu ra (mặc định: 10).
        """
        super().__init__()

        self.in_channels = in_channels
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes

        # Khối tích chập 1: 3x32x32 -> conv1_channels x 16 x 16
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Khối tích chập 2: conv1_channels x 16 x 16 -> conv2_channels x 8 x 8
        self.conv2 = nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Kích thước phẳng sau 2 lần MaxPool: conv2_channels * 8 * 8
        self.flattened_size = conv2_channels * 8 * 8
        self.flatten = nn.Flatten()

        # Khối phân loại (Classifier)
        self.fc1 = nn.Linear(self.flattened_size, fc_hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward Pass).

        Args:
            x: Tensor đầu vào shape [B, 3, 32, 32].

        Returns:
            torch.Tensor: Tensor logits thô shape [B, 10].
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Trải phẳng và phân loại
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """
        Trả về siêu dữ liệu cấu hình kiến trúc để lưu trong checkpoint.
        """
        return {
            "architecture": "CIFAR10CNN",
            "in_channels": self.in_channels,
            "conv1_channels": self.conv1_channels,
            "conv2_channels": self.conv2_channels,
            "fc_hidden_dim": self.fc_hidden_dim,
            "num_classes": self.num_classes,
        }
