"""
Module định nghĩa kiến trúc mạng nơ-ron truyền thẳng (Multi-Layer Perceptron - MLP)
cho bài toán phân loại ảnh FashionMNIST.

KIẾN TRÚC MÔ HÌNH:
1. Flatten layer: Chuyển ma trận ảnh 2D [B, 1, 28, 28] thành vector đặc trưng 1D [B, 784] (1 * 28 * 28 = 784).
2. Linear (Fully-Connected) 1: 784 -> hidden_dim1 (mặc định: 128).
3. Hàm kích hoạt phi tuyến tính (Activation): ReLU (Rectified Linear Unit: f(x) = max(0, x)).
4. Linear 2: hidden_dim1 -> hidden_dim2 (mặc định: 64).
5. ReLU: f(x) = max(0, x).
6. Linear 3 (Output Layer): hidden_dim2 -> 10 (Số lượng lớp cần phân loại).

TẠI SAO KHÔNG DÙNG SOFTMAX TRONG MÔ HÌNH KHI SỬ DỤNG CrossEntropyLoss?
- Trong PyTorch, hàm mất mát `nn.CrossEntropyLoss()` đã tích hợp sẵn bên trong hai bước tính toán:
    1. Log-Softmax: `log(exp(x_i) / sum(exp(x_j)))`
    2. NLLLoss (Negative Log Likelihood Loss)
- Việc kết hợp hai bước này bằng kỹ thuật tính toán số học Log-Sum-Exp giúp ngăn ngừa hiện tượng
  tràn số (overflow) hoặc tắt dần giá trị về 0 (underflow) của hàm mũ e^x khi giá trị logit quá lớn hoặc quá nhỏ.
- Nếu bạn thêm `nn.Softmax()` vào cuối mô hình rồi truyền kết quả đó vào `CrossEntropyLoss`,
  PyTorch sẽ tính `log(softmax(softmax(x)))`, dẫn tới sai lệch toán học nghiêm trọng và gradient bị suy biến.
- Do đó: Mô hình LUÔN trả về Logits thô (raw unnormalized scores). Khi cần xác suất (probabilities)
  để hiển thị cho con người đọc, ta chỉ cần gọi `torch.softmax(logits, dim=-1)` ở bước dự đoán (predict).
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class FashionMNISTMLP(nn.Module):
    """
    Mạng Perceptron đa tầng (MLP) cho phân loại ảnh FashionMNIST.
    """

    def __init__(
        self,
        input_features: int = 28 * 28,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 10,
    ) -> None:
        """
        Khởi tạo kiến trúc MLP.

        Args:
            input_features: Số lượng đặc trưng đầu vào (1 * 28 * 28 = 784).
            hidden_dim1: Số nơ-ron của tầng ẩn thứ nhất (mặc định 128).
            hidden_dim2: Số nơ-ron của tầng ẩn thứ hai (mặc định 64).
            num_classes: Số lượng lớp phân loại ở đầu ra (mặc định 10).
        """
        super().__init__()

        self.input_features = input_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes

        # Khai báo các tầng mạng
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward Pass).

        Args:
            x: Tensor đầu vào có kích thước [Batch_Size, 1, 28, 28] hoặc [Batch_Size, 784].

        Returns:
            torch.Tensor: Tensor logits thô có kích thước [Batch_Size, num_classes].
        """
        # Trải phẳng tensor ảnh: [B, 1, 28, 28] -> [B, 784]
        out = self.flatten(x)

        # Tầng ẩn 1
        out = self.fc1(out)
        out = self.relu1(out)

        # Tầng ẩn 2
        out = self.fc2(out)
        out = self.relu2(out)

        # Tầng đầu ra (Logits thô, chưa qua Softmax)
        logits = self.fc3(out)
        return logits

    def get_config(self) -> Dict[str, Any]:
        """
        Trả về cấu hình siêu tham số của kiến trúc mô hình để lưu vào checkpoint.
        """
        return {
            "architecture": "FashionMNISTMLP",
            "input_features": self.input_features,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "num_classes": self.num_classes,
        }
