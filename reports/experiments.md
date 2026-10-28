# Báo cáo Thực nghiệm Mô hình (Experiments Report)

> **Lưu ý khoa học quan trọng:**
> 1. Mọi so sánh và lựa chọn siêu tham số (hyperparameters) dưới đây đều được thực hiện dựa trên **tập Validation**, giữ độc lập hoàn toàn với **tập Test**.
> 2. **Không so sánh độ chính xác của FashionMNIST với CIFAR-10**: Đây là hai bài toán hoàn toàn khác biệt về độ khó. FashionMNIST là ảnh thang độ xám (grayscale, 1 kênh, nền đen) với hình dáng vật thể rõ ràng; còn CIFAR-10 là ảnh màu tự nhiên (RGB, 3 kênh, nền phức tạp, góc chụp đa dạng). Việc một mô hình đạt 86% trên FashionMNIST trong khi đạt 58% trên CIFAR-10 không có nghĩa là mô hình trên tốt hơn mô hình dưới.

---

## 1. Môi trường thực nghiệm

- **Hệ điều hành**: Windows 11 (win32)
- **Python**: 3.14.3
- **PyTorch**: 2.14.1+cpu
- **TorchVision**: 0.29.1+cpu
- **Thiết bị tính toán**: CPU
- **Random seed**: 42 (áp dụng đồng nhất cho việc chia train/val và khởi tạo trọng số)

---

## 2. Thực nghiệm Bài toán A — FashionMNIST + MLP

### 2.1. Cấu hình kiểm soát (Controlled Variables)

Trong các thực nghiệm này, ta thay đổi **từng biến số một tại một thời điểm** (One-Variable-At-A-Time):
- Cố định: `batch_size = 128`, `optimizer = Adam`, `val_size = 10,000`, `seed = 42`.
- Biến số kiểm tra:
  1. **Baseline**: `epochs = 2`, `lr = 0.001`, `hidden = 128 -> 64`.
  2. **Thực nghiệm LR**: Thay đổi `lr = 0.01` (tăng gấp 10 lần), giữ nguyên cấu trúc mạng và số epoch.
  3. **Thực nghiệm Hidden Width**: Tăng độ rộng tầng ẩn `256 -> 128` (tăng số tham số từ 109k lên 235k), giữ nguyên `lr = 0.001`.

### 2.2. Bảng kết quả đo đạc thực tế

| Mã thực nghiệm | Kiến trúc ẩn | Learning Rate | Epochs | Train Loss (Ep 2) | Train Acc (Ep 2) | Val Loss (Ep 2) | Val Acc (Ep 2) | Thời gian huấn luyện |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-Base** | 128 -> 64 (109k params) | 0.001 | 2 | 0.4306 | 84.76% | 0.4066 | **86.00%** | 18.72 s |
| **F-LR01** | 128 -> 64 (109k params) | 0.01 | 2 | 0.3995 | 85.45% | 0.3772 | **86.83%** | 15.80 s |
| **F-Width** | 256 -> 128 (235k params) | 0.001 | 2 | 0.4022 | 85.53% | 0.3777 | **86.41%** | 19.09 s |

### 2.3. Quan sát và Đánh giá (Observations)

1. **Ảnh hưởng của Learning Rate (0.001 vs 0.01)**:
   - Với Adam và kiến trúc MLP nông (shallow), tăng `lr` từ 0.001 lên 0.01 giúp mô hình hội tụ nhanh hơn trong 2 epoch đầu (Train Acc đạt 85.45% so với 84.76%, Val Acc đạt 86.83% so với 86.00%).
   - Tuy nhiên, trong Deep Learning, LR quá lớn có thể khiến mô hình bị dao động hoặc phân kỳ nếu huấn luyện nhiều epoch hơn.
2. **Ảnh hưởng của độ rộng tầng ẩn (Width: 128->64 vs 256->128)**:
   - Tăng số nơ-ron giúp tăng dung lượng biểu diễn (capacity) của mô hình (từ 109,386 tham số lên 235,146 tham số).
   - Val Acc tăng nhẹ từ 86.00% lên 86.41%, Train Loss giảm sâu hơn (0.4022 so với 0.4306).
   - Thời gian tính toán tăng thêm khoảng 0.37s do kích thước phép nhân ma trận lớn hơn.

---

## 3. Thực nghiệm Bài toán B — CIFAR-10 + CNN

### 3.1. Cấu hình kiểm soát (Controlled Variables)

- Cố định: `batch_size = 128`, `optimizer = Adam`, `lr = 0.001`, `val_size = 10,000`, `seed = 42`.
- Biến số kiểm tra:
  1. **Baseline**: `Conv1 = 32`, `Conv2 = 64`, `FC = 128`, `Augmentation = False` (545,098 tham số).
  2. **Thực nghiệm Augmentation**: Bật Data Augmentation (RandomCrop 32 padding 4 + RandomHorizontalFlip 50%), giữ nguyên các thông số khác.
  3. **Thực nghiệm Kênh Tích chập (Channels)**: Giảm kích thước bộ lọc xuống `Conv1 = 16`, `Conv2 = 32` (268,650 tham số), tắt Augmentation.

### 3.2. Bảng kết quả đo đạc thực tế

| Mã thực nghiệm | Kênh Conv | Data Augmentation | Epochs | Train Loss | Train Acc | Val Loss | Val Acc | Thời gian chạy (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C-Base** | 32 -> 64 (545k params) | Tắt | 1 | 1.4292 | 48.89% | 1.1448 | **58.77%** | 32.02 s |
| **C-Augment**| 32 -> 64 (545k params) | **Bật (Crop + Flip)** | 1 | 1.6515 | 39.66% | 1.3077 | **52.74%** | 38.77 s |
| **C-Small** | 16 -> 32 (268k params) | Tắt | 1 | 1.5083 | 45.91% | 1.2714 | **54.72%** | 23.32 s |

### 3.3. Quan sát và Đánh giá (Observations)

1. **Hiện tượng của Data Augmentation ở các epoch đầu**:
   - Khi bật Data Augmentation, Train Acc ở epoch 1 giảm xuống 39.66% (so với 48.89% của Baseline) và Val Acc đạt 52.74% (so với 58.77%).
   - **Giải thích nguyên lý**: Data Augmentation làm biến đổi ngẫu nhiên từng ảnh huấn luyện tại mỗi bước (cắt xén viền, lật ngang). Điều này làm tăng độ khó của tập dữ liệu huấn luyện, ngăn chặn việc mô hình học vẹt (overfitting). Trong các mạng nơ-ron thị giác, Data Augmentation đòi hỏi mô hình cần nhiều epoch hơn (thường là 15-50 epochs) để hội tụ, nhưng khi hội tụ sẽ cho khả năng tổng quát hóa (generalization) cao hơn hẳn mô hình không có augmentation.
   - Thời gian chạy tăng từ 32s lên 38.77s do chi phí biến đổi ảnh ngẫu nhiên trong CPU worker.
2. **Ảnh hưởng của số lượng kênh tích chập (Capacity & Speed)**:
   - Thu nhỏ mô hình (giảm một nửa số kênh tích chập) cắt giảm hơn 50% số tham số (từ 545,098 xuống 268,650 tham số).
   - Tốc độ huấn luyện tăng rõ rệt: giảm từ 32.02s xuống **23.32s** trên CPU.
   - Đổi lại, độ chính xác trên tập Validation giảm từ 58.77% xuống 54.72% do mô hình nhỏ hơn có ít bộ lọc không gian để nắm bắt các đặc trưng vi mô của ảnh màu.

---

## 4. Kết luận rút ra cho học viên IT

1. **Quy tắc thay đổi biến số**: Chỉ thay đổi đúng một nhân tố tại một thời điểm để biết chính xác sự thay đổi của Loss và Accuracy đến từ đâu.
2. **Kỷ luật dữ liệu**: Tập Test chỉ được mở ra đánh giá đúng một lần sau cùng. Mọi quyết định tinh chỉnh tham số (chọn model checkpoint nào, chọn lr nào, chọn kênh nào) bắt buộc phải căn cứ trên **tập Validation**.
3. **Hiểu bản chất phần cứng và chi phí tính toán**: Mạng CNN xử lý dữ liệu 3D tensor (C, H, W) qua các phép tích chập trượt (sliding convolution) tốn nhiều phép tính hơn đáng kể so với việc trải phẳng ảnh thành vector 1D của MLP.
