"""
Module xử lý dữ liệu FashionMNIST.

TẬP DỮ LIỆU FASHION-MNIST:
- Gồm 70,000 ảnh grayscale (ảnh đơn kênh xám) kích thước 28x28 pixels.
- 60,000 ảnh tập huấn luyện chính thức (Official Train set).
- 10,000 ảnh tập kiểm tra chính thức (Official Test set).
- 10 lớp thời trang của Zalando (chỉ số nhãn từ 0 đến 9):
    0: T-shirt/top (Áo thun)
    1: Trouser     (Quần dài)
    2: Pullover    (Áo len chui đầu)
    3: Dress       (Váy liền thân)
    4: Coat        (Áo khoác)
    5: Sandal      (Dép quai hậu)
    6: Shirt       (Áo sơ mi)
    7: Sneaker     (Giày thể thao)
    8: Bag         (Túi xách)
    9: Ankle boot  (Ủng/bốt cổ thấp)

CHIẾN LƯỢC TIỀN XỬ LÝ VÀ PHÂN CHIA (PREPROCESSING & SPLITTING STRATEGY):
1. Biến đổi (Transform): Sử dụng `transforms.ToTensor()`:
   - Chuyển ảnh PIL (0-255) thành Tensor kiểu `torch.float32`.
   - Chuẩn hóa giá trị pixel về khoảng [0.0, 1.0].
   - Đổi thứ tự chiều từ HWC sang CHW: Shape cuối cùng là [1, 28, 28].
2. Phân chia Train / Validation:
   - Chia 60,000 ảnh huấn luyện chính thức thành:
     * 50,000 ảnh cho tập Train (dùng để cập nhật gradient).
     * 10,000 ảnh cho tập Validation (dùng để theo dõi overfitting và chọn best model).
   - Sử dụng `torch.Generator().manual_seed(seed)` để việc chia tập luôn nhất quán giữa các lần chạy.
3. Tập Test (10,000 ảnh):
   - Giữ nguyên độc lập tuyệt đối, chỉ dùng để đánh giá mô hình cuối cùng sau khi hoàn tất huấn luyện.
"""

from typing import Tuple, List
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 10 lớp phân loại của FashionMNIST
FASHION_MNIST_CLASSES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_fashion_mnist_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    val_size: int = 10000,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tải tập dữ liệu FashionMNIST và tạo DataLoader cho train, validation, và test.

    Args:
        data_dir: Thư mục lưu trữ dữ liệu tải về (mặc định 'data').
        batch_size: Kích thước mỗi batch dữ liệu (mặc định 64).
        val_size: Số lượng ảnh trích xuất từ train set làm tập validation (mặc định 10,000).
        num_workers: Số luồng nạp dữ liệu đa tiến trình (mặc định 0 để tương thích tốt nhất trên Windows).
        seed: Seed cố định cho phép tách tập ngẫu nhiên có thể tái lập.

    Returns:
        tuple (train_loader, val_loader, test_loader, class_names):
            - train_loader: DataLoader cho tập huấn luyện (được xáo trộn shuffle=True).
            - val_loader: DataLoader cho tập xác thực (shuffle=False).
            - test_loader: DataLoader cho tập kiểm tra độc lập (shuffle=False).
            - class_names: Danh sách 10 tên lớp.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Chuẩn hóa ảnh: Đưa về Tensor [0.0, 1.0] và shape [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Tải toàn bộ tập train chính thức (60,000 ảnh)
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Tải tập test chính thức độc lập (10,000 ảnh)
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Phân chia train/validation
    total_train = len(full_train_dataset)
    train_size = total_train - val_size
    assert train_size > 0, f"val_size ({val_size}) phải nhỏ hơn tổng số ảnh train ({total_train})"

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, FASHION_MNIST_CLASSES
