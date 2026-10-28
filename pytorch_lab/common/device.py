"""
Module quản lý thiết bị tính toán (Compute Device).

Trong PyTorch, mọi Tensor và Mô hình (nn.Module) đều nằm trên một thiết bị cụ thể:
- 'cpu': Bộ xử lý trung tâm của máy tính.
- 'cuda' hoặc 'cuda:0': Bộ xử lý đồ họa (GPU NVIDIA) thông qua nền tảng CUDA.

Việc tách biệt rõ ràng việc chọn thiết bị giúp mã nguồn linh hoạt:
chạy được trên máy tính phổ thông không có GPU NVIDIA, đồng thời tự động tận dụng
GPU mạnh mẽ khi có sẵn mà không cần thay đổi logic chương trình.
"""

from typing import Dict, Any
import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Xác định và trả về thiết bị tính toán phù hợp nhất.

    Nguyên tắc hoạt động:
    1. Kiểm tra xem người dùng có muốn dùng CUDA không (`prefer_cuda=True`).
    2. Kiểm tra xem PyTorch có phát hiện driver và phần cứng CUDA hợp lệ không (`torch.cuda.is_available()`).
    3. Nếu thỏa mãn cả 2 điều kiện, trả về `torch.device('cuda')`.
    4. Ngược lại, trả về `torch.device('cpu')`.

    Args:
        prefer_cuda: Ưu tiên dùng CUDA nếu có sẵn (mặc định: True).

    Returns:
        torch.device: Đối tượng thiết bị ('cuda' hoặc 'cpu').
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Thu thập thông tin chi tiết về phần cứng tính toán hiện tại.

    Returns:
        dict: Chứa các thông tin:
            - 'cuda_available': bool (CUDA có khả dụng không)
            - 'device_count': int (Số lượng GPU CUDA phát hiện được)
            - 'device_name': str (Tên card GPU hoặc 'CPU')
            - 'selected_device': str (Tên thiết bị mặc định được chọn)
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "CPU"
    selected = str(get_device())

    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "selected_device": selected,
    }
