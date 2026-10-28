"""
Tập lệnh dự đoán mẫu ảnh đơn từ tập kiểm tra CIFAR-10.

Quy trình:
1. Nạp checkpoint mô hình CNN.
2. Lấy một mẫu ảnh từ test dataset (kèm tiền xử lý chuẩn hóa CIFAR-10).
3. Lan truyền xuôi qua CNN để nhận logits.
4. Chuyển đổi thành xác suất phần trăm qua `torch.softmax()`.
5. Hiển thị nhãn thực tế, nhãn dự đoán, độ tin cậy, và Top-3 lớp có khả năng cao nhất.

Cách chạy:
    python -m pytorch_lab.cifar10.predict
    python -m pytorch_lab.cifar10.predict --sample-idx 7
"""

import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import torch
from torchvision import datasets, transforms

from pytorch_lab.common.device import get_device
from pytorch_lab.common.checkpoint import load_checkpoint
from pytorch_lab.cifar10.model import CIFAR10CNN
from pytorch_lab.cifar10.data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán nhãn cho một mẫu ảnh CIFAR-10 từ tập kiểm tra."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/cifar10/best_model.pt",
        help="Đường dẫn file checkpoint mô hình",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Chỉ số mẫu ảnh trong tập kiểm tra (0 đến 9999, mặc định: 0)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Thư mục chứa dataset")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng lớp có xác suất cao nhất cần hiển thị")
    return parser.parse_args()


def predict(args: argparse.Namespace) -> None:
    device = get_device()

    print("=" * 65)
    print("      DỰ ĐOÁN MẪU ẢNH CIFAR-10 TỪ TẬP TEST")
    print("=" * 65)

    # 1. Tải tập test kèm transform chuẩn hóa
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    if args.sample_idx < 0 or args.sample_idx >= len(test_dataset):
        raise ValueError(
            f"sample-idx phải nằm trong khoảng [0, {len(test_dataset) - 1}], nhưng nhận được: {args.sample_idx}"
        )

    image, true_label = test_dataset[args.sample_idx]
    image_batch = image.unsqueeze(0).to(device)

    # 2. Khởi tạo và nạp trọng số mô hình
    model = CIFAR10CNN()
    checkpoint = load_checkpoint(
        filepath=args.checkpoint_path,
        model=model,
        device=device,
    )
    class_names = checkpoint.get("class_names") or CIFAR10_CLASSES

    # 3. Suy diễn
    model.eval()
    with torch.inference_mode():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    # 4. Top-K
    top_k = min(args.top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    pred_class_idx = top_indices[0].item()
    pred_prob = top_probs[0].item()

    true_class_name = class_names[true_label]
    pred_class_name = class_names[pred_class_idx]
    is_correct = pred_class_idx == true_label

    # 5. In kết quả trực quan
    print(f"[*] Mẫu kiểm tra số        : #{args.sample_idx}")
    print(f"[*] Kích thước Tensor ảnh   : {list(image.shape)} (Kênh C=3 RGB, Cao H=32, Rộng W=32)")
    print(f"[*] Nhãn thực tế (Ground Truth) : {true_label} ({true_class_name})")
    print(f"[*] Nhãn dự đoán (Prediction)   : {pred_class_idx} ({pred_class_name})")
    print(f"[*] Độ tin cậy (Confidence)     : {pred_prob * 100:.2f}%")
    print(f"[*] Kết quả kiểm tra            : {'CHÍNH XÁC [PASS]' if is_correct else 'SAI LỆCH [FAIL]'}")
    print("-" * 65)
    print(f"TOP-{top_k} LỚP DỰ ĐOÁN CÓ XÁC SUẤT CAO NHẤT:")
    for rank in range(top_k):
        idx = top_indices[rank].item()
        prob = top_probs[rank].item()
        name = class_names[idx]
        bar = "#" * int(prob * 30)
        print(f"  {rank + 1}. [{idx}] {name:<15}: {prob * 100:>6.2f}% | {bar}")
    print("=" * 65)


def main() -> None:
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    main()
