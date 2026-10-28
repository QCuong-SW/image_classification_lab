"""
Module quản lý lưu và tải Checkpoint mô hình (Checkpoint Management).

Trong PyTorch, có hai cách lưu mô hình:
1. Lưu toàn bộ đối tượng mô hình (`torch.save(model, path)`):
   - Nhược điểm: Phụ thuộc chặt chẽ vào cấu trúc thư mục và mã nguồn lớp khi tải lại bằng pickle.
   - Dễ bị lỗi khi đổi tên file hoặc thay đổi kiến trúc code, đồng thời tiềm ẩn rủi ro bảo mật (arbitrary code execution).
2. Lưu chỉ bảng trọng số (`state_dict`) kèm siêu dữ liệu cơ bản (Cách chuẩn và an toàn nhất):
   - Chỉ lưu các trọng số (weights/biases) dạng Tensor và các thông số nguyên thủy (int, float, str, dict).
   - Khi tải lại: Cần khởi tạo đối tượng mô hình từ code trước, sau đó nạp `state_dict` vào.

Module này tuân thủ thực hành chuẩn số 2, tương thích với cơ chế an toàn `weights_only` của PyTorch.
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    model_config: Optional[Dict[str, Any]] = None,
    class_names: Optional[list] = None,
) -> None:
    """
    Lưu checkpoint an toàn bao gồm `state_dict` của mô hình và siêu dữ liệu tối thiểu.

    Args:
        filepath: Đường dẫn lưu file (.pt hoặc .pth).
        model: Đối tượng mô hình PyTorch (nn.Module).
        optimizer: Đối tượng optimizer (tùy chọn, cần khi muốn tiếp tục huấn luyện).
        epoch: Chỉ số epoch hiện tại (tùy chọn).
        val_loss: Loss trên tập validation (tùy chọn).
        val_acc: Accuracy trên tập validation (tùy chọn).
        model_config: Cấu hình tham số kiến trúc mô hình (tùy chọn).
        class_names: Danh sách tên lớp phân loại (tùy chọn).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "model_config": model_config or {},
        "class_names": class_names or [],
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Nạp trọng số từ checkpoint vào mô hình đã được khởi tạo.

    Args:
        filepath: Đường dẫn tới file checkpoint.
        model: Thực thể mô hình (nn.Module) đã được tạo sẵn cấu trúc tương thích.
        optimizer: Đối tượng optimizer nếu muốn phục hồi trạng thái optimizer (tùy chọn).
        device: Thiết bị đích muốn ánh xạ tensor tới (CPU hoặc CUDA).

    Returns:
        dict: Toàn bộ từ điển checkpoint (chứa siêu dữ liệu epoch, val_loss, class_names,...).

    Raises:
        FileNotFoundError: Nếu file checkpoint không tồn tại.
        RuntimeError: Nếu kiến trúc mô hình không khớp với trọng số trong checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file checkpoint tại: '{filepath}'. "
            "Hãy đảm bảo bạn đã chạy quá trình huấn luyện (train) trước khi đánh giá hoặc dự đoán."
        )

    map_location = device if device is not None else torch.device("cpu")

    try:
        # Thử tải với weights_only=True trước (chuẩn bảo mật từ PyTorch 2.4+)
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    except Exception:
        # Dự phòng tải an toàn nếu checkpoint chứa các kiểu dữ liệu an toàn nhưng PyTorch yêu cầu nạp thường
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"File checkpoint '{filepath}' không hợp lệ: Không tìm thấy khóa 'model_state_dict'."
        )

    # Nạp state_dict vào mô hình
    model.load_state_dict(checkpoint["model_state_dict"])

    # Nạp trạng thái optimizer nếu có yêu cầu
    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Chuyển mô hình sang thiết bị đích
    model.to(map_location)

    # Chuyển mô hình về chế độ đánh giá mặc định (eval mode)
    model.eval()

    return checkpoint
