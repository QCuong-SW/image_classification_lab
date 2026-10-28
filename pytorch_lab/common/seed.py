"""
Module quản lý hạt giống ngẫu nhiên (Random Seed) cho tính tái lập (Reproducibility).

Tại sao cần cố định Random Seed trong Deep Learning?
- Khởi tạo trọng số ngẫu nhiên của mô hình (Weights initialization).
- Xáo trộn dữ liệu (Shuffle) trong DataLoader trước mỗi epoch.
- Phân chia tập dữ liệu train/validation (Random train-validation split).
- Các phép biến đổi dữ liệu tăng cường (Data augmentation như RandomCrop, RandomHorizontalFlip).

Nếu không cố định seed, mỗi lần bạn chạy lại code, kết quả loss và accuracy sẽ thay đổi nhẹ,
khiến việc so sánh hiệu quả giữa các siêu tham số (hyperparameters) trở nên thiếu tin cậy.

LƯU Ý QUAN TRỌNG VỀ TÍNH KHÁCH QUAN (DETERMINISM):
Ngay cả khi đã cố định seed, tính xác định tuyệt đối (100% deterministic) trên GPU
vẫn có thể bị ảnh hưởng bởi các thuật toán tối ưu hóa phần cứng (như cuDNN benchmark)
hoặc các phép toán cộng dồn gradient không đồng bộ (atomic operations). Do đó, kết quả
có thể có sai số rất nhỏ giữa các kiến trúc phần cứng khác nhau.
"""

import os
import random
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Cố định hạt giống ngẫu nhiên cho Python, NumPy (nếu có), và PyTorch (CPU + CUDA).

    Args:
        seed: Số nguyên dùng làm hạt giống (mặc định: 42).
        deterministic: Nếu True, yêu cầu cuDNN sử dụng các thuật toán tất định (deterministic).
                       Điều này có thể làm giảm hiệu năng huấn luyện trên GPU nhưng tăng độ tái lập.
    """
    # 1. Cố định biến môi trường băm của Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Cố định seed cho module random chuẩn của Python
    random.seed(seed)

    # 3. Cố định seed cho NumPy nếu thư viện này được cài đặt
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # 4. Cố định seed cho PyTorch trên CPU
    torch.manual_seed(seed)

    # 5. Cố định seed cho tất cả các GPU CUDA (nếu có)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 6. Thiết lập chế độ thuật toán cuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Benchmark=True cho phép cuDNN tự tìm thuật toán convolution nhanh nhất cho kích thước ảnh hiện tại
        torch.backends.cudnn.benchmark = True
