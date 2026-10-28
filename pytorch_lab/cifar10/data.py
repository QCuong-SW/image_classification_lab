"""
Module xử lý dữ liệu CIFAR-10.

TẬP DỮ LIỆU CIFAR-10:
- Gồm 60,000 ảnh màu RGB kích thước 32x32 pixels thuộc 10 lớp vật thể tự nhiên:
    0: airplane   (Máy bay)
    1: automobile (Ô tô)
    2: bird       (Chim chóc)
    3: cat        (Mèo)
    4: deer       (Hươu/Nai)
    5: dog        (Chó)
    6: frog       (Ếch)
    7: horse      (Ngựa)
    8: ship       (Tàu thủy)
    9: truck      (Xe tải)
- 50,000 ảnh huấn luyện (Train set).
- 10,000 ảnh kiểm tra độc lập (Test set).

HÌNH DẠNG TENSOR (TENSOR SHAPE):
- Mỗi ảnh có kích thước: [Kênh C=3, Cao H=32, Rộng W=32].
- Khi nạp qua DataLoader với batch_size: Tensor có shape [B, 3, 32, 32].

CHIẾN LƯỢC TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION):
- Khi `augment=True` (chỉ áp dụng cho tập Train):
    + `RandomCrop(32, padding=4)`: Cắt ngẫu nhiên ảnh kèm thêm viền đệm 4 pixel để mô hình học tính bất biến vị trí (translation invariance).
    + `RandomHorizontalFlip()`: Lật ngang ảnh ngẫu nhiên 50% để mô hình hiểu rằng con mèo quay sang trái hay phải vẫn là con mèo.
- Tập Validation và Test: TUYỆT ĐỐI KHÔNG dùng phép biến đổi ngẫu nhiên (chỉ dùng ToTensor và Normalize)
  để kết quả đánh giá luôn khách quan và xác định.
"""

from typing import Tuple, List, Optional
import os
import tarfile
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


# 10 lớp phân loại của CIFAR-10
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Giá trị trung bình và độ lệch chuẩn chuẩn hóa chuẩn mực cho CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Danh sách URLs dự phòng phòng trường hợp server cs.toronto.edu bị bóp băng thông
MIRROR_URLS = [
    "https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]


def ensure_cifar10_downloaded(data_dir: str) -> None:
    """
    Tự động tải và giải nén CIFAR-10 từ mirror tốc độ cao nếu thư mục chưa tồn tại.
    """
    extracted_folder = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.exists(extracted_folder):
        return

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    downloaded = False
    for url in MIRROR_URLS:
        try:
            print(f"[*] Đang tải CIFAR-10 từ mirror: {url} ...")
            urllib.request.urlretrieve(url, tar_path)
            downloaded = True
            break
        except Exception:
            continue

    if not downloaded:
        # Để torchvision tự thử lại nếu cả 2 mirror trên thất bại
        return

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
    except Exception as e:
        print(f"[-] Cảnh báo giải nén: {e}")


class TransformedSubset(Dataset):
    """
    Lớp bọc (Wrapper) áp dụng phép biến đổi riêng cho từng tập con sau khi phân chia train/validation,
    đảm bảo Data Augmentation chỉ kích hoạt trên tập Train mà không làm biến dạng tập Validation.
    """

    def __init__(self, subset: Dataset, transform: Optional[transforms.Compose] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, target = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_cifar10_transforms(augment: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Tạo bộ biến đổi dữ liệu cho tập Train và Validation/Test.

    Args:
        augment: Có kích hoạt tăng cường dữ liệu cho tập train hay không.

    Returns:
        tuple (train_transform, eval_transform)
    """
    # Phép biến đổi chuẩn tắc cho Validation và Test (không có yếu tố ngẫu nhiên)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def get_cifar10_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    val_size: int = 10000,
    augment: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tải tập dữ liệu CIFAR-10 và chuẩn bị DataLoader cho train, validation, và test.

    Args:
        data_dir: Thư mục lưu trữ dữ liệu tải về.
        batch_size: Kích thước batch.
        val_size: Số lượng ảnh trích xuất từ 50,000 ảnh train làm tập validation.
        augment: Kích hoạt tăng cường dữ liệu cho tập train (Data Augmentation).
        num_workers: Số luồng tải dữ liệu (mặc định 0 trên Windows).
        seed: Seed cố định cho việc phân chia train/validation.

    Returns:
        tuple (train_loader, val_loader, test_loader, class_names)
    """
    os.makedirs(data_dir, exist_ok=True)
    ensure_cifar10_downloaded(data_dir)

    train_transform, eval_transform = get_cifar10_transforms(augment=augment)

    # Tải tập dữ liệu gốc dạng ảnh PIL (chưa áp dụng transform tại đây để gán transform riêng sau khi chia)
    raw_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    raw_test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None,
    )

    # Phân chia train/val từ 50,000 ảnh gốc
    total_train = len(raw_train_dataset)
    train_size = total_train - val_size
    assert train_size > 0, f"val_size ({val_size}) phải nhỏ hơn tổng số ảnh train ({total_train})"

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Áp dụng transform độc lập
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=eval_transform)
    test_dataset = TransformedSubset(raw_test_dataset, transform=eval_transform)

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

    return train_loader, val_loader, test_loader, CIFAR10_CLASSES
