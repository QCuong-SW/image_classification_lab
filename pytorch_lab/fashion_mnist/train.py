"""
Tập lệnh huấn luyện mô hình MLP trên tập dữ liệu FashionMNIST.

Các bước thực hiện:
1. Đọc tham số dòng lệnh (CLI arguments) với `argparse`.
2. Cố định seed cho tính tái lập.
3. Khởi tạo DataLoader cho tập Train và Validation (tập Test được giữ nguyên độc lập).
4. Khởi tạo mô hình `FashionMNISTMLP`, hàm mất mát `nn.CrossEntropyLoss`, và thuật toán tối ưu `Adam`.
5. Vòng lặp huấn luyện qua từng epoch:
   - Gọi `train_one_epoch` cập nhật trọng số.
   - Gọi `evaluate_one_epoch` đánh giá trên tập Validation (không cập nhật tham số).
   - So sánh kết quả validation accuracy để lưu lại checkpoint tốt nhất (Best Checkpoint).

Cách chạy:
    python -m pytorch_lab.fashion_mnist.train
    python -m pytorch_lab.fashion_mnist.train --epochs 5 --lr 0.001 --batch-size 64
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
from pytorch_lab.fashion_mnist.data import get_fashion_mnist_dataloaders
from pytorch_lab.fashion_mnist.model import FashionMNISTMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình MLP phân loại thời trang FashionMNIST."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch huấn luyện (mặc định: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học Learning Rate (mặc định: 0.001)")
    parser.add_argument("--hidden-dim1", type=int, default=128, help="Số nơ-ron tầng ẩn 1 (mặc định: 128)")
    parser.add_argument("--hidden-dim2", type=int, default=64, help="Số nơ-ron tầng ẩn 2 (mặc định: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--data-dir", type=str, default="data", help="Thư mục lưu trữ dataset (mặc định: data)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/fashion_mnist",
        help="Thư mục lưu checkpoint mô hình (mặc định: checkpoints/fashion_mnist)",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    # 1. Thiết lập seed và thiết bị
    set_seed(args.seed)
    device = get_device()

    print("=" * 72)
    print("      HUẤN LUYỆN FASHION-MNIST VỚI KIẾN TRÚC MẠNG MLP")
    print("=" * 72)
    print(f"[*] Thiết bị tính toán  : {device}")
    print(f"[*] Số epochs           : {args.epochs}")
    print(f"[*] Batch size          : {args.batch_size}")
    print(f"[*] Learning rate (lr)  : {args.lr}")
    print(f"[*] Tầng ẩn (Hidden dims): {args.hidden_dim1} -> {args.hidden_dim2}")
    print(f"[*] Random seed         : {args.seed}")
    print("-" * 72)

    # 2. Tải dữ liệu
    print("[*] Đang nạp dữ liệu FashionMNIST...")
    train_loader, val_loader, _, class_names = get_fashion_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=10000,
        seed=args.seed,
    )
    print(f"[+] Đã tải xong: {len(train_loader.dataset)} mẫu train, {len(val_loader.dataset)} mẫu validation.")

    # 3. Khởi tạo mô hình
    model = FashionMNISTMLP(
        input_features=28 * 28,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        num_classes=len(class_names),
    )
    model.to(device)

    # Đếm số lượng tham số học được
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Tổng số tham số mô hình (Trainable Parameters): {total_params:,}")

    # 4. Thiết lập Loss function và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Vòng lặp huấn luyện chính
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

        # Huấn luyện 1 epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Đánh giá trên tập validation
        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration = time.time() - epoch_start

        # Kiểm tra lưu checkpoint tốt nhất
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

    # Lưu checkpoint của epoch cuối cùng
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
