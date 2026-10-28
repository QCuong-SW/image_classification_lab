"""
Module tính toán các chỉ số đánh giá (Evaluation Metrics).

Đối với bài toán phân loại đa lớp (Multi-class classification):
1. Overall Accuracy (Độ chính xác tổng thể): Tỷ lệ số mẫu dự đoán đúng trên tổng số mẫu.
2. Per-class Accuracy (Độ chính xác theo từng lớp): Đánh giá mô hình nhận diện tốt/kém lớp nào.
3. Confusion Matrix (Ma trận nhầm lẫn): Ma trận kích thước [C, C] trong đó hàng (row) là
   nhãn thực tế (ground truth) và cột (column) là nhãn dự đoán (prediction).
   - Đường chéo chính (diagonal) biểu thị các mẫu được dự đoán chính xác.
   - Các phần tử ngoài đường chéo cho biết lớp nào thường bị mô hình nhận nhầm thành lớp nào.
"""

from typing import List, Dict, Any, Union
import torch


def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Tính độ chính xác tổng thể (Overall Accuracy).

    Args:
        y_pred: Tensor chứa logits dự đoán [batch_size, num_classes]
                hoặc chỉ số lớp đã argmax [batch_size].
        y_true: Tensor chứa nhãn thực tế [batch_size].

    Returns:
        float: Giá trị độ chính xác trong khoảng [0.0, 1.0].
    """
    if y_pred.ndim > 1:
        # Nếu đầu vào là raw logits hoặc probabilities, lấy chỉ số lớp có giá trị lớn nhất
        predictions = torch.argmax(y_pred, dim=-1)
    else:
        predictions = y_pred

    correct = (predictions == y_true).sum().item()
    total = y_true.numel()
    return correct / total if total > 0 else 0.0


def confusion_matrix(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Tạo ma trận nhầm lẫn (Confusion Matrix) không phụ thuộc scikit-learn.

    Hàng i: Lớp thực tế (Ground Truth) là i.
    Cột j: Mô hình dự đoán là j.
    Giá trị cm[i, j]: Số lượng mẫu lớp i bị đoán thành lớp j.

    Args:
        y_pred: Tensor nhãn dự đoán [N] (kiểu int/long).
        y_true: Tensor nhãn thực tế [N] (kiểu int/long).
        num_classes: Tổng số lớp.

    Returns:
        torch.Tensor: Ma trận [num_classes, num_classes] kiểu torch.int64.
    """
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=-1)

    y_pred = y_pred.view(-1).to(torch.int64)
    y_true = y_true.view(-1).to(torch.int64)

    # Sử dụng công thức bincount hiệu năng cao:
    # index = y_true * num_classes + y_pred
    indices = y_true * num_classes + y_pred
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def calculate_per_class_accuracy(cm: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Tính độ chính xác cho từng lớp dựa trên ma trận nhầm lẫn.

    Args:
        cm: Ma trận nhầm lẫn [num_classes, num_classes].

    Returns:
        list of dict: Mỗi phần tử chứa thông tin {'class_idx', 'correct', 'total', 'accuracy'}.
    """
    num_classes = cm.shape[0]
    results = []

    for c in range(num_classes):
        total = cm[c].sum().item()
        correct = cm[c, c].item()
        acc = (correct / total) if total > 0 else 0.0
        results.append({
            "class_idx": c,
            "correct": correct,
            "total": total,
            "accuracy": acc,
        })
    return results


def print_evaluation_report(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    class_names: List[str]
) -> None:
    """
    In báo cáo đánh giá chi tiết ra console (Overall Accuracy, Per-Class, Confusion Matrix).

    Args:
        y_pred: Logits dự đoán hoặc chỉ số lớp [N, C] hoặc [N].
        y_true: Nhãn thực tế [N].
        class_names: Danh sách tên hiển thị của các lớp.
    """
    if y_pred.ndim > 1:
        y_pred_classes = torch.argmax(y_pred, dim=-1)
    else:
        y_pred_classes = y_pred

    num_classes = len(class_names)
    overall_acc = calculate_accuracy(y_pred_classes, y_true)
    cm = confusion_matrix(y_pred_classes, y_true, num_classes=num_classes)
    per_class = calculate_per_class_accuracy(cm)

    print("\n" + "=" * 70)
    print("                     BÁO CÁO ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)
    print(f"[*] Độ chính xác tổng thể (Overall Accuracy): {overall_acc * 100:.2f}% "
          f"({(y_pred_classes == y_true).sum().item()}/{y_true.numel()} mẫu)")
    print("-" * 70)
    print(f"{'Class ID':<10} {'Tên Lớp (Class Name)':<25} {'Đúng/Tổng':<15} {'Độ chính xác':<15}")
    print("-" * 70)

    for item in per_class:
        idx = item["class_idx"]
        name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        counts = f"{item['correct']}/{item['total']}"
        acc_pct = f"{item['accuracy'] * 100:.2f}%"
        print(f"{idx:<10} {name:<25} {counts:<15} {acc_pct:<15}")

    print("-" * 70)
    print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("(Hàng = Nhãn thực tế, Cột = Nhãn dự đoán)")
    print()

    # In tiêu đề cột
    header = f"{'Thực\\Đoán':<12}" + "".join(f"[{i:^4}]" for i in range(num_classes))
    print(header)
    print("-" * len(header))

    # In từng dòng ma trận
    for i in range(num_classes):
        row_str = f"[{i:^2}] {class_names[i][:8]:<6}: "
        row_vals = "".join(f"{cm[i, j].item():^6}" for j in range(num_classes))
        print(row_str + row_vals)

    print("=" * 70 + "\n")
