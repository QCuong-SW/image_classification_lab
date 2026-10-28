"""
Stage 1 — Hướng dẫn học Tensor và Cơ chế Autograd từ gốc rễ (First Principles).

Module này độc lập với việc huấn luyện mô hình ảnh, được thiết kế dưới dạng
bài giảng tương tác console dành cho sinh viên IT chuyển từ NumPy/scikit-learn sang PyTorch.

Nội dung bao gồm 8 phần:
1. Khởi tạo Tensor (Tensor Creation).
2. Các thuộc tính cốt lõi: shape, ndim, dtype, device.
3. Truy cập và cắt lát (Indexing & Slicing).
4. Biến đổi hình dạng và trải phẳng (Reshaping & Flattening).
5. Phép toán từng phần tử (Elementwise) vs Nhân ma trận (Matrix Multiplication).
6. Khái niệm theo dõi gradient với `requires_grad=True`.
7. Xây dựng đồ thị tính toán và gọi `backward()`.
8. Đọc giá trị đạo hàm riêng (.grad) và ý nghĩa trong Gradient Descent.

Cách chạy:
    python -m pytorch_lab.tensors_tutorial
"""

import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch


def section_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_tensor_creation() -> None:
    section_header("1. KHỞI TẠO TENSOR (TENSOR CREATION)")
    print("Tensor là cấu trúc dữ liệu trung tâm của PyTorch, tương tự ndarray của NumPy")
    print("nhưng được tối ưu hóa để chạy trên GPU và có khả năng tự động tính đạo hàm.\n")

    # Từ danh sách Python
    t_from_list = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    print("a) Khởi tạo từ list Python:")
    print(f"   torch.tensor([1, 2, 3, 4]) -> {t_from_list}")

    # Tensor toàn số 0 (zeros) và toàn số 1 (ones)
    t_zeros = torch.zeros(size=(2, 3), dtype=torch.float32)
    t_ones = torch.ones(size=(2, 3), dtype=torch.float32)
    print("\nb) Tensor toàn số 0 (zeros) kích thước 2 hàng, 3 cột:")
    print(f"   {t_zeros}")
    print("\nc) Tensor toàn số 1 (ones) kích thước 2 hàng, 3 cột:")
    print(f"   {t_ones}")

    # Dãy số cách đều
    t_arange = torch.arange(start=0, end=10, step=2)
    print(f"\nd) Dãy số cách đều (arange 0 đến 10 bước 2): {t_arange}")

    # Khởi tạo ngẫu nhiên theo phân phối chuẩn chuẩn hóa (Mean=0, Std=1)
    t_randn = torch.randn(size=(2, 2))
    print(f"\ne) Khởi tạo ngẫu nhiên (randn 2x2):\n{t_randn}")


def demo_tensor_properties() -> None:
    section_header("2. BỐN THUỘC TÍNH CỐT LÕI CỦA TENSOR (SHAPE, NDIM, DTYPE, DEVICE)")
    
    t = torch.randn(size=(3, 4, 5), dtype=torch.float32)
    print(f"Tạo tensor t với khởi tạo ngẫu nhiên:")
    print(f"- t.shape  : {t.shape} -> Kích thước các chiều: 3 khối, mỗi khối 4 hàng, 5 cột.")
    print(f"- t.ndim   : {t.ndim} -> Số chiều (Rank/Dimensions). Ở đây là 3D tensor.")
    print(f"- t.dtype  : {t.dtype} -> Kiểu dữ liệu phần tử (ví dụ: float32, int64, bool).")
    print(f"- t.device : {t.device} -> Thiết bị lưu trữ tensor trong RAM (cpu) hoặc VRAM (cuda:0).")
    print("\n* Lưu ý cho người mới: Trong Deep Learning phân loại ảnh, tensor đầu vào")
    print("  thường có dạng 4D: (Batch_Size, Channels, Height, Width) viết tắt là (B, C, H, W).")


def demo_indexing_slicing() -> None:
    section_header("3. TRUY CẬP VÀ CẮT LÁT TENSOR (INDEXING & SLICING)")
    
    data = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ])
    print(f"Ma trận gốc kích thước {data.shape}:\n{data}\n")

    # Lấy phần tử đơn
    val = data[1, 2].item()
    print(f"- Lấy phần tử tại hàng 1, cột 2 (0-indexed): data[1, 2] = {val}")

    # Cắt một hàng hoặc một cột
    row_0 = data[0, :]
    col_1 = data[:, 1]
    print(f"- Lấy toàn bộ hàng 0: data[0, :] = {row_0}")
    print(f"- Lấy toàn bộ cột 1: data[:, 1] = {col_1}")

    # Lọc theo điều kiện logic (Boolean Masking)
    mask = data > 60
    filtered = data[mask]
    print(f"- Mặt nạ logic data > 60:\n{mask}")
    print(f"- Các phần tử thỏa mãn > 60: {filtered}")


def demo_reshaping_flattening() -> None:
    section_header("4. BIẾN ĐỔI HÌNH DẠNG VÀ TRẢI PHẲNG (RESHAPING & FLATTENING)")
    print("Mạng MLP (Multi-Layer Perceptron) chỉ nhận đầu vào là vector 1D (features).")
    print("Do đó, một ảnh 2D (ví dụ 1x28x28) bắt buộc phải được trải phẳng thành 784 giá trị.\n")

    img_batch = torch.arange(1, 25, dtype=torch.float32).reshape(2, 1, 3, 4)
    print(f"Batch ảnh giả lập kích thước (B=2, C=1, H=3, W=4): shape = {img_batch.shape}")

    # Cách 1: Sử dụng .view() hoặc .reshape()
    # Kích thước -1 nghĩa là PyTorch sẽ tự động tính toán chiều còn lại sao cho tổng số phần tử không đổi
    flattened_view = img_batch.view(2, -1)
    print(f"- Sau khi biến đổi qua .view(2, -1): shape = {flattened_view.shape}")
    print(f"  Giải thích: 1 * 3 * 4 = 12 đặc trưng cho mỗi ảnh trong batch.")

    # Cách 2: Sử dụng torch.flatten()
    flattened_builtin = torch.flatten(img_batch, start_dim=1)
    print(f"- Dùng torch.flatten(img_batch, start_dim=1): shape = {flattened_builtin.shape}")
    print("  (Giữ nguyên chiều Batch=2, trải phẳng toàn bộ các chiều sau: 1*3*4 = 12).")


def demo_elementwise_and_matmul() -> None:
    section_header("5. PHÉP TOÁN TỪNG PHẦN TỬ VS NHÂN MA TRẬN (ELEMENTWISE VS MATMUL)")
    
    A = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0],
                      [7.0, 8.0]])

    print(f"Ma trận A (2x2):\n{A}")
    print(f"Ma trận B (2x2):\n{B}\n")

    # 1. Phép nhân từng phần tử tương ứng (Hadamard product)
    elem_mul = A * B
    print(f"a) Phép nhân từng phần tử (A * B):\n{elem_mul}")
    print("   Giải thích: Vị trí (0,0) là 1*5=5; Vị trí (0,1) là 2*6=12.")

    # 2. Phép nhân ma trận đại số tuyến tính (Matrix Multiplication)
    mat_mul = torch.matmul(A, B)  # Hoặc viết tắt là A @ B
    print(f"\nb) Phép nhân ma trận đại số tuyến tính (A @ B):\n{mat_mul}")
    print("   Giải thích: Hàng 0 của A dot Cột 0 của B: 1*5 + 2*7 = 5 + 14 = 19.")
    print("   QUY TẮC HÌNH DẠNG: Ma trận (M x K) @ (K x N) -> Ma trận (M x N).")
    print("   Số cột của ma trận trước PHẢI BẰNG số hàng của ma trận sau.")


def demo_autograd_basic() -> None:
    section_header("6. CƠ CHẾ AUTOGRAD & requires_grad=True")
    print("Khi tạo một Tensor với requires_grad=True, PyTorch sẽ bắt đầu ghi nhận")
    print("mọi phép toán thực hiện trên tensor đó để xây dựng Đồ thị tính toán (DAG - Directed Acyclic Graph).")
    print("Nhờ đó, PyTorch có thể tự động tính đạo hàm ngược bằng thuật toán Backpropagation.\n")

    x = torch.tensor(2.0, requires_grad=True)
    print(f"Khởi tạo scalar tensor: x = {x.item()}, requires_grad={x.requires_grad}")

    # Giả sử hàm số: y = 3 * x^2 + 2 * x + 1
    # Đạo hàm giải tích: dy/dx = 6 * x + 2
    # Với x = 2.0: dy/dx = 6 * (2.0) + 2 = 14.0
    y = 3 * (x ** 2) + 2 * x + 1
    print(f"Tính giá trị biểu thức: y = 3*x^2 + 2*x + 1 = {y.item()}")
    print(f"Thuộc tính grad_fn của y: {y.grad_fn} (Ghi nhận phép toán tạo ra y)")

    # Gọi backward() để lan truyền ngược đạo hàm
    y.backward()

    print(f"\nSau khi gọi y.backward():")
    print(f"Đạo hàm dy/dx được lưu tại x.grad: {x.grad.item()}")
    print(f"Kiểm chứng toán học: 6 * (2.0) + 2 = 14.0 -> CHÍNH XÁC TUYỆT ĐỐI!")


def demo_autograd_multivariable() -> None:
    section_header("7. ĐẠO HÀM RIÊNG ĐA BIẾN (MULTIVARIABLE AUTOGRAD)")
    print("Trong mạng Neural, mô hình có hàng triệu tham số W (weights) và b (biases).")
    print("Loss là một hàm số phụ thuộc vào tất cả các tham số này.\n")

    # Hai biến độc lập: a và b
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(4.0, requires_grad=True)

    # Hàm mất mát giả định: L = 2 * a^2 + b^3
    # Đạo hàm riêng theo a: dL/da = 4 * a = 4 * 3.0 = 12.0
    # Đạo hàm riêng theo b: dL/db = 3 * b^2 = 3 * (4.0^2) = 48.0
    L = 2 * (a ** 2) + (b ** 3)
    print(f"Biểu thức: L = 2*a^2 + b^3 với a=3.0, b=4.0 -> L = {L.item()}")

    # Lan truyền ngược
    L.backward()

    print(f"- Đạo hàm riêng dL/da (a.grad): {a.grad.item()} (Kỳ vọng: 4*3 = 12.0)")
    print(f"- Đạo hàm riêng dL/db (b.grad): {b.grad.item()} (Kỳ vọng: 3*16 = 48.0)")


def demo_gradient_descent_concept() -> None:
    section_header("8. Ý NGHĨA GRADIENT TRONG TỐI ƯU HÓA (GRADIENT DESCENT)")
    print("Công thức cập nhật trọng số cơ bản nhất của Machine Learning / Deep Learning:")
    print("        W_moi = W_cu - learning_rate * W.grad\n")
    print("Dấu trừ (-) biểu thị việc di chuyển ngược hướng dốc (gradient) để đi về điểm cực tiểu của Loss.")
    print("LƯU Ý QUAN TRỌNG:")
    print("Trong PyTorch, mặc định gradient sẽ CỘNG DỒN (accumulate) sau mỗi lần gọi backward().")
    print("Do đó, trong vòng lặp huấn luyện, BẮT BUỘC phải gọi 'optimizer.zero_grad()' trước mỗi batch")
    print("để xóa sạch gradient cũ trước khi tính gradient mới!")
    print("=" * 70)


def main() -> None:
    print("\n" + "#" * 70)
    print("   PYTORCH FIRST-PRINCIPLES: TENSOR VÀ AUTOGRAD TUTORIAL")
    print("   Dành cho sinh viên IT nắm bắt bản chất tính toán Deep Learning")
    print("#" * 70)

    demo_tensor_creation()
    demo_tensor_properties()
    demo_indexing_slicing()
    demo_reshaping_flattening()
    demo_elementwise_and_matmul()
    demo_autograd_basic()
    demo_autograd_multivariable()
    demo_gradient_descent_concept()

    print("\n[+] Bài học Tensor & Autograd đã hoàn tất thành công!\n")


if __name__ == "__main__":
    main()
