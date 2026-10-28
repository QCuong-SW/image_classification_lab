"""
Module động cơ huấn luyện và đánh giá (Training & Evaluation Engine).

Module này cung cấp hai hàm cốt lõi của quy trình Deep Learning:
1. `train_one_epoch`: Duyệt qua toàn bộ dữ liệu huấn luyện một lần, thực hiện lan truyền xuôi
   (Forward pass), tính hàm mất mát (Loss), lan truyền ngược (Backward pass / Backpropagation),
   và cập nhật trọng số mô hình bằng Optimizer.
2. `evaluate_one_epoch`: Duyệt qua tập Validation hoặc Test, tính toán Loss và Accuracy
   ở chế độ suy diễn (`torch.inference_mode()`) và đảm bảo tuyệt đối KHÔNG cập nhật tham số.

CÁCH TÍNH LOSS VÀ ACCURACY CHUẨN XÁC:
- Không lấy trung bình cộng đơn thuần của các batch accuracy (vì batch cuối cùng có thể có số mẫu ít hơn).
- Luôn cộng dồn tổng loss (`loss.item() * batch_size`) và tổng số mẫu dự đoán đúng (`correct`),
  sau đó chia cho tổng số mẫu thực tế của toàn bộ dataset (`total_samples`).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Huấn luyện mô hình qua 1 epoch hoàn chỉnh trên DataLoader.

    Quy trình từng bước cho mỗi batch:
    1. `model.train()`: Kích hoạt chế độ huấn luyện (ảnh hưởng tới Dropout, BatchNorm nếu có).
    2. Chuyển inputs và labels lên thiết bị tính toán (CPU hoặc CUDA).
    3. `optimizer.zero_grad()`: Xóa sạch gradient tích lũy từ batch trước.
    4. Forward pass: `outputs = model(inputs)`. outputs có kích thước [batch_size, num_classes].
    5. Tính hàm mất mát: `loss = criterion(outputs, labels)`.
    6. Lan truyền ngược (Backpropagation): `loss.backward()` tính đạo hàm riêng (gradient)
       của hàm mất mát theo từng tham số mô hình.
    7. Cập nhật trọng số: `optimizer.step()` dịch chuyển tham số ngược hướng gradient theo learning rate.
    8. Cộng dồn số liệu thống kê (Loss và số lượng dự đoán đúng).

    Args:
        model: Mô hình PyTorch (nn.Module).
        dataloader: DataLoader cung cấp các batch dữ liệu (inputs, labels).
        criterion: Hàm mất mát (thường là nn.CrossEntropyLoss).
        optimizer: Thuật toán tối ưu hóa (ví dụ: torch.optim.Adam hoặc SGD).
        device: Thiết bị thực thi (torch.device('cuda') hoặc torch.device('cpu')).

    Returns:
        tuple (epoch_loss, epoch_accuracy):
            - epoch_loss (float): Giá trị hàm mất mát trung bình trên toàn bộ tập dữ liệu.
            - epoch_accuracy (float): Độ chính xác trung bình [0.0, 1.0].
    """
    # 1. Chuyển mô hình sang chế độ huấn luyện
    model.train()
    model.to(device)

    running_loss: float = 0.0
    correct_samples: int = 0
    total_samples: int = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 2. Đưa dữ liệu lên thiết bị đích (GPU hoặc CPU)
        # inputs shape:  [B, C, H, W]
        # targets shape: [B]
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = targets.size(0)

        # 3. Xóa gradient cũ trước khi tính mới
        optimizer.zero_grad()

        # 4. Lan truyền xuôi (Forward pass)
        # outputs shape: [B, num_classes] (raw logits)
        outputs = model(inputs)

        # 5. Tính giá trị hàm mất mát
        loss = criterion(outputs, targets)

        # 6. Lan truyền ngược (Backward pass)
        loss.backward()

        # 7. Cập nhật tham số mô hình
        optimizer.step()

        # 8. Thu thập số liệu thống kê chính xác
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(outputs, dim=-1)
        correct_samples += (predictions == targets).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_samples / total_samples if total_samples > 0 else 0.0

    return epoch_loss, epoch_acc


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Đánh giá mô hình trên một tập dữ liệu (Validation hoặc Test) mà KHÔNG cập nhật tham số.

    Điểm khác biệt trọng yếu so với hàm huấn luyện:
    1. `model.eval()`: Tắt các hành vi ngẫu nhiên khi huấn luyện (như Dropout).
    2. `torch.inference_mode()` (hoặc `torch.no_grad()`): Vô hiệu hóa Autograd engine,
       không lưu đồ thị tính toán (computation graph), giúp tiết kiệm bộ nhớ RAM/VRAM và tăng tốc độ.
    3. Không gọi `loss.backward()` hay `optimizer.step()`.

    Args:
        model: Mô hình PyTorch (nn.Module).
        dataloader: DataLoader chứa dữ liệu đánh giá.
        criterion: Hàm mất mát (nn.CrossEntropyLoss).
        device: Thiết bị tính toán.

    Returns:
        tuple (eval_loss, eval_accuracy):
            - eval_loss (float): Loss trung bình trên tập đánh giá.
            - eval_accuracy (float): Độ chính xác trung bình [0.0, 1.0].
    """
    # 1. Chuyển mô hình sang chế độ đánh giá
    model.eval()
    model.to(device)

    running_loss: float = 0.0
    correct_samples: int = 0
    total_samples: int = 0

    # 2. Ngắt cơ chế theo dõi gradient để tăng tốc và tiết kiệm bộ nhớ
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = targets.size(0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Cộng dồn
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=-1)
            correct_samples += (predictions == targets).sum().item()
            total_samples += batch_size

    eval_loss = running_loss / total_samples if total_samples > 0 else 0.0
    eval_acc = correct_samples / total_samples if total_samples > 0 else 0.0

    return eval_loss, eval_acc
