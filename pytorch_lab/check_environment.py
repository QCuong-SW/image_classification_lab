"""
Stage 0 — Kiểm tra môi trường thực thi (Environment Diagnostic).

Tập lệnh này kiểm tra:
1. Phiên bản Python đang sử dụng.
2. Phiên bản PyTorch và TorchVision đã cài đặt.
3. Trạng thái hỗ trợ phần cứng tăng tốc đồ họa (CUDA / NVIDIA GPU).
4. Thiết bị tính toán được hệ thống tự động lựa chọn (CPU hoặc GPU).
5. Thực hiện một phép tính Tensor cơ bản trên thiết bị đã chọn để xác minh khả năng tính toán.

Cách chạy từ thư mục gốc của dự án:
    python -m pytorch_lab.check_environment
"""

import sys

# Đảm bảo in Unicode tiếng Việt mượt mà trên console Windows mà không bị lỗi cp1252
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torchvision

from pytorch_lab.common.device import get_device, get_device_info


def main() -> None:
    print("=" * 65)
    print("      PYTORCH IMAGE LAB — KIỂM TRA MÔI TRƯỜNG THỰC THI")
    print("=" * 65)

    # 1. Thông tin phiên bản phần mềm
    print(f"[*] Python Version     : {sys.version.split()[0]} ({sys.platform})")
    print(f"[*] PyTorch Version    : {torch.__version__}")
    print(f"[*] TorchVision Version: {torchvision.__version__}")

    # 2. Thông tin thiết bị phần cứng
    info = get_device_info()
    cuda_available = info["cuda_available"]
    print(f"[*] CUDA Available     : {'CO (San sang)' if cuda_available else 'KHONG (Chua bat hoac dung ban CPU)'}")

    if cuda_available:
        print(f"[*] So luong GPU       : {info['device_count']}")
        print(f"[*] Ten GPU phat hien  : {info['device_name']}")
        print(f"[*] CUDA Version PyTorch: {torch.version.cuda}")
    else:
        print("[*] Ghi chu GPU        : PyTorch hien dang chay che do CPU.")
        print("                         (Neu ban co card NVIDIA, hay cai ban PyTorch ho tro CUDA)")

    # 3. Thiết bị được chỉ định
    device = get_device()
    print(f"[*] Thiet bi duoc chon : {device}")

    # 4. Phép tính Tensor kiểm chứng trên thiết bị đã chọn
    print("-" * 65)
    print(f"[*] Thuc hien phep tinh thu nghiem tren thiet bi [{device}]...")
    
    # Tạo 2 ma trận ngẫu nhiên 3x3 trên thiết bị
    matrix_a = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]], device=device)
    matrix_b = torch.tensor([[9.0, 8.0, 7.0],
                             [6.0, 5.0, 4.0],
                             [3.0, 2.0, 1.0]], device=device)

    # Phép nhân ma trận (Matrix Multiplication)
    result = torch.matmul(matrix_a, matrix_b)

    print(f"    - Ma tran A shape: {matrix_a.shape}, dtype: {matrix_a.dtype}, device: {matrix_a.device}")
    print(f"    - Ma tran B shape: {matrix_b.shape}, dtype: {matrix_b.dtype}, device: {matrix_b.device}")
    print(f"    - Ket qua A @ B  :\n{result}")
    print("-" * 65)
    print("[+] Kiem tra moi truong HOAN TAT THANH CONG!")
    print("=" * 65)


if __name__ == "__main__":
    main()
