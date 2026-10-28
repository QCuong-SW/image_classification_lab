"""
Tập lệnh đánh giá mô hình FashionMNIST trên tập dữ liệu TEST độc lập.

TẬP TEST ĐỘC LẬP:
- Tập test hoàn toàn không tham gia vào quá trình cập nhật gradient (train)
  cũng như quá trình chọn mô hình tốt nhất (validation).
- Việc đánh giá trên tập test phản ánh khách quan năng lực tổng quát hóa (Generalization)
  của mô hình trên dữ liệu thực tế chưa từng thấy trước đây.

Cách chạy:
    python -m pytorch_lab.fashion_mnist.evaluate
    python -m pytorch_lab.fashion_mnist.evaluate --checkpoint-path checkpoints/fashion_mnist/best_model.pt
"""

import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import torch
import torch.nn as nn

from pytorch_lab.common.device import get_device
from pytorch_lab.common.checkpoint import load_checkpoint
from pytorch_lab.common.metrics import print_evaluation_report
from pytorch_lab.fashion_mnist.data import get_fashion_mnist_dataloaders
from pytorch_lab.fashion_mnist.model import FashionMNISTMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình FashionMNIST trên tập kiểm tra độc lập (Test Set)."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/fashion_mnist/best_model.pt",
        help="Đường dẫn tới file checkpoint mô hình đã huấn luyện",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--data-dir", type=str, default="data", help="Thư mục chứa dataset (mặc định: data)")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    print("=" * 70)
    print("      ĐÁNH GIÁ MÔ HÌNH FASHION-MNIST TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 70)
    print(f"[*] Thiết bị tính toán: {device}")
    print(f"[*] File checkpoint    : {args.checkpoint_path}")

    # 1. Nạp tập test độc lập
    _, _, test_loader, default_class_names = get_fashion_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"[*] Đã nạp tập kiểm tra Test Set: {len(test_loader.dataset)} ảnh.")

    # 2. Khởi tạo cấu trúc mô hình
    # Tạm tạo mô hình với cấu hình mặc định, load_checkpoint sẽ đọc cấu hình nếu có
    model = FashionMNISTMLP()

    # 3. Nạp trọng số từ checkpoint
    checkpoint = load_checkpoint(
        filepath=args.checkpoint_path,
        model=model,
        device=device,
    )
    class_names = checkpoint.get("class_names") or default_class_names
    saved_val_acc = checkpoint.get("val_acc")
    if saved_val_acc is not None:
        print(f"[*] Độ chính xác Validation tại checkpoint đã lưu: {saved_val_acc * 100:.2f}%")

    # 4. Dự đoán toàn bộ tập test
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    print("[*] Đang thực hiện suy diễn (Inference) trên tập Test...")
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            total_samples += batch_size

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    avg_test_loss = total_loss / total_samples

    print(f"[+] Hoàn tất! Test Loss trung bình: {avg_test_loss:.4f}")

    # 5. In báo cáo đánh giá chi tiết
    print_evaluation_report(
        y_pred=y_pred,
        y_true=y_true,
        class_names=class_names,
    )


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
