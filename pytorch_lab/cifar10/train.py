"""
Tập lệnh huấn luyện mô hình mạng nơ-ron tích chập (CNN) trên tập dữ liệu CIFAR-10.

Các bước thực hiện:
1. Đọc tham số cấu hình qua `argparse` (epochs, lr, batch_size, augment, channels).
2. Thiết lập seed và chọn thiết bị (CPU/CUDA).
3. Khởi tạo DataLoader tập Train và Validation. Nếu có cờ `--augment`, kích hoạt data augmentation cho tập train.
4. Khởi tạo mô hình `CIFAR10CNN`, Loss `CrossEntropyLoss`, và Optimizer `Adam`.
5. Huấn luyện qua từng epoch, đánh giá validation, và lưu checkpoint tốt nhất.

Cách chạy:
    python -m pytorch_lab.cifar10.train
    python -m pytorch_lab.cifar10.train --epochs 5 --lr 0.001 --augment
"""

import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lab.common.device import get_device
from pytorch_lab.common.seed import set_seed
from pytorch_lab.common.engine import train_one_epoch, evaluate_one_epoch
from pytorch_lab.common.checkpoint import save_checkpoint
from pytorch_lab.cifar10.data import get_cifar10_dataloaders
from pytorch_lab.cifar10.model import CIFAR10CNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình CNN phân loại ảnh màu CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện (mặc định: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học Learning Rate (mặc định: 0.001)")
    parser.add_argument("--conv1-channels", type=int, default=32, help="Số kênh Conv1 (mặc định: 32)")
    parser.add_argument("--conv2-channels", type=int, default=64, help="Số kênh Conv2 (mặc định: 64)")
    parser.add_argument("--fc-hidden-dim", type=int, default=128, help="Số nơ-ron tầng ẩn FC (mặc định: 128)")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Bật tính năng tăng cường dữ liệu cho tập huấn luyện (Data Augmentation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--data-dir", type=str, default="data", help="Thư mục dataset (mặc định: data)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/cifar10",
        help="Thư mục lưu checkpoint (mặc định: checkpoints/cifar10)",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()

    print("=" * 72)
    print("      HUẤN LUYỆN CIFAR-10 VỚI KIẾN TRÚC MẠNG TÍCH CHẬP CNN")
    print("=" * 72)
    print(f"[*] Thiết bị tính toán  : {device}")
    print(f"[*] Số epochs           : {args.epochs}")
    print(f"[*] Batch size          : {args.batch_size}")
    print(f"[*] Learning rate (lr)  : {args.lr}")
    print(f"[*] Cấu hình kênh Conv  : Conv1={args.conv1_channels}, Conv2={args.conv2_channels}, FC={args.fc_hidden_dim}")
    print(f"[*] Data Augmentation   : {'BẬT (RandomCrop + Flip)' if args.augment else 'TẮT (Chỉ chuẩn hóa)'}")
    print(f"[*] Random seed         : {args.seed}")
    print("-" * 72)

    # 1. Tải dữ liệu
    print("[*] Đang nạp dữ liệu CIFAR-10...")
    train_loader, val_loader, _, class_names = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=10000,
        augment=args.augment,
        seed=args.seed,
    )
    print(f"[+] Đã tải xong: {len(train_loader.dataset)} mẫu train, {len(val_loader.dataset)} mẫu validation.")

    # 2. Khởi tạo mô hình
    model = CIFAR10CNN(
        in_channels=3,
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        fc_hidden_dim=args.fc_hidden_dim,
        num_classes=len(class_names),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Tổng số tham số mô hình (Trainable Parameters): {total_params:,}")

    # 3. Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Checkpoints
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(args.checkpoint_dir, "last_model.pt")

    print("\nBẮT ĐẦU VÒNG LẶP HUẤN LUYỆN:")
    print("-" * 72)
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<14} {'Thời gian':<10}")
    print("-" * 72)

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration = time.time() - epoch_start

        is_best = val_acc > best_val_acc
        status_flag = ""
        if is_best:
            best_val_acc = val_acc
            status_flag = "(*) BEST"
            save_checkpoint(
                filepath=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                model_config=model.get_config(),
                class_names=class_names,
            )

        print(
            f"{epoch:<8} {train_loss:<14.4f} {train_acc * 100:<13.2f}% "
            f"{val_loss:<14.4f} {val_acc * 100:<13.2f}% {epoch_duration:<7.2f}s {status_flag}"
        )

    # Lưu checkpoint epoch cuối
    save_checkpoint(
        filepath=last_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        val_loss=val_loss,
        val_acc=val_acc,
        model_config=model.get_config(),
        class_names=class_names,
    )

    total_duration = time.time() - total_start_time
    print("-" * 72)
    print(f"[+] Huấn luyện hoàn thành trong {total_duration:.2f} giây.")
    print(f"[+] Checkpoint tốt nhất được lưu tại : {best_checkpoint_path} (Val Acc: {best_val_acc * 100:.2f}%)")
    print(f"[+] Checkpoint cuối cùng được lưu tại : {last_checkpoint_path}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
