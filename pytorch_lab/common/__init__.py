"""
Các tiện ích dùng chung (Common utilities) cho pytorch_lab:
- device: Quản lý thiết bị tính toán (CPU / CUDA)
- seed: Cố định seed cho tính tái lập kết quả (reproducibility)
- engine: Vòng lặp huấn luyện (train) và đánh giá (evaluate) chuẩn
- checkpoint: Lưu trữ và phục hồi mô hình an toàn
- metrics: Tính toán độ chính xác và ma trận nhầm lẫn (confusion matrix)
"""

from pytorch_lab.common.device import get_device, get_device_info
from pytorch_lab.common.seed import set_seed
from pytorch_lab.common.engine import train_one_epoch, evaluate_one_epoch
from pytorch_lab.common.checkpoint import save_checkpoint, load_checkpoint
from pytorch_lab.common.metrics import calculate_accuracy, calculate_per_class_accuracy, confusion_matrix, print_evaluation_report

__all__ = [
    "get_device",
    "get_device_info",
    "set_seed",
    "train_one_epoch",
    "evaluate_one_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "calculate_accuracy",
    "calculate_per_class_accuracy",
    "confusion_matrix",
    "print_evaluation_report",
]
