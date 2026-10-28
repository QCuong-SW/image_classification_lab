# PyTorch Image Classification Lab (pytorch-image-lab)

Kho tài liệu và mã nguồn thực hành PyTorch và Deep Learning từ bản chất toán học và tính toán (First Principles), dành cho sinh viên CNTT đã có nền tảng về Python, scikit-learn, và các khái niệm Machine Learning cơ bản.

Dự án này tuân thủ nguyên tắc: **Console-only** (giao diện dòng lệnh thuần túy), không sử dụng web frontend, không API service cồng kềnh, không notebook bị ẩn giấu trạng thái thực thi. Mọi dòng code đều tường minh, dễ đọc và trực tiếp.

---

## Mục lục

1. [Mục tiêu và Triết lý của Dự án](#1-mục-tiêu-và-triết-lý-của-dự-án)
2. [Tại sao có 2 bài toán thực nghiệm? (FashionMNIST vs CIFAR-10)](#2-tại-sao-có-2-bài-toán-thực-nghiệm)
3. [Cấu trúc Thư mục và Trách nhiệm từng File](#3-cấu-trúc-thư-mục-và-trách-nhiệm-từng-file)
4. [Cài đặt và Xác minh Môi trường](#4-cài-đặt-và-xác-minh-môi-trường)
5. [Stage 1 — Học Tensor và Autograd từ Gốc](#5-stage-1--học-tensor-và-autograd-từ-gốc)
6. [Hướng dẫn Lệnh Thực thi (Train, Evaluate, Predict)](#6-hướng-dẫn-lệnh-thực-thi-train-evaluate-predict)
7. [Hình dạng Dữ liệu (Shapes) và Bảng Nhãn Lớp](#7-hình-dạng-dữ-liệu-shapes-và-bảng-nhãn-lớp)
8. [Phân tích Vòng lặp Huấn luyện (Training Loop Step-by-Step)](#8-phân-tích-vòng-lặp-huấn-luyện-step-by-step)
9. [Sự khác biệt giữa Validation và Test Evaluation](#9-sự-khác-biệt-giữa-validation-và-test-evaluation)
10. [Cách kiểm tra Checkpoint đã lưu](#10-cách-kiểm-tra-checkpoint-đã-lưu)
11. [Các lỗi thường gặp và Cách khắc phục (Troubleshooting)](#11-các-lỗi-thường-gặp-và-cách-khắc-phục)
12. [Giới hạn Thực nghiệm và Hướng phát triển](#12-giới-hạn-thực-nghiệm-và-hướng-phát-triển)

---

## 1. Mục tiêu và Triết lý của Dự án

Trong scikit-learn, bạn thường chỉ cần gọi `model.fit(X, y)` và `model.predict(X)`. Việc này rất tiện lợi nhưng vô tình biến quá trình tối ưu hóa thành một "hộp đen" (black box). 

Dự án này mở chiếc hộp đen đó ra bằng PyTorch:
- Tự tay điều khiển từng bước lan truyền xuôi (**Forward pass**), tính hàm mất mát (**Loss**), xóa gradient cũ (**Zero grad**), lan truyền ngược (**Backpropagation / Autograd**), và cập nhật trọng số (**Optimizer step**).
- Hiểu rõ việc chuyển đổi hình dạng tensor qua từng tầng: từ ảnh 2D/3D thành vector 1D, hoặc qua các bộ lọc tích chập (kernels) và pooling.
- Rèn luyện tư duy thực nghiệm khoa học: Cố định seed, phân chia dữ liệu Train/Val/Test độc lập, và đánh giá ma trận nhầm lẫn (Confusion Matrix).

---

## 2. Tại sao có 2 bài toán thực nghiệm?

Dự án cung cấp 2 bài toán kinh điển tương ứng với 2 kiến trúc nền tảng của Deep Learning:

| Tiêu chí | Bài toán A: FashionMNIST | Bài toán B: CIFAR-10 |
| :--- | :--- | :--- |
| **Kiến trúc mạng** | **MLP** (Multi-Layer Perceptron) | **CNN** (Convolutional Neural Network) |
| **Đặc điểm dữ liệu** | Ảnh thang độ xám (1 kênh), kích thước 28x28 | Ảnh màu RGB (3 kênh), kích thước 32x32 |
| **Cách xử lý ảnh** | Trải phẳng (Flatten) toàn bộ 784 pixels thành vector 1D | Dùng bộ lọc không gian (Conv2d 3x3) trượt quét đặc trưng |
| **Bảo toàn không gian** | Bị mất thông tin vị trí lân cận giữa các pixel | Bảo toàn mối tương quan không gian (Spatial locality) |
| **Độ phức tạp** | Nhẹ, chạy tốt ngay cả trên CPU yếu | Nặng hơn, đòi hỏi tích chập và pooling nhiều tầng |

> **Cảnh báo khoa học:** Không so sánh độ chính xác của FashionMNIST trực tiếp với CIFAR-10 để kết luận mô hình nào "thông minh" hơn. CIFAR-10 là bài toán khó hơn gấp nhiều lần về mặt thị giác máy tính.

---

## 3. Cấu trúc Thư mục và Trách nhiệm từng File

```text
pytorch-image-lab/
├── README.md                      # Tài liệu hướng dẫn toàn diện (file này)
├── requirements.txt               # Danh sách thư viện tối thiểu cần thiết
├── .gitignore                     # Bỏ qua dataset, checkpoint, cache
├── download_cifar10.py (tùy chọn) # Tự động tải từ mirror tốc độ cao
├── pytorch_lab/
│   ├── __init__.py                # Package init
│   ├── check_environment.py       # Stage 0: Chẩn đoán môi trường phần cứng/PyTorch
│   ├── tensors_tutorial.py        # Stage 1: Bài giảng Tensor & Autograd tương tác console
│   ├── common/                    # Các module dùng chung
│   │   ├── __init__.py
│   │   ├── device.py              # Helper tự động nhận diện CPU / CUDA
│   │   ├── seed.py                # Cố định Random Seed cho tính tái lập
│   │   ├── engine.py              # Vòng lặp train_one_epoch và evaluate_one_epoch
│   │   ├── checkpoint.py          # Lưu/tải state_dict và metadata an toàn
│   │   └── metrics.py             # Accuracy, per-class accuracy, Confusion Matrix
│   ├── fashion_mnist/             # Bài toán A: Phân loại thời trang với MLP
│   │   ├── __init__.py
│   │   ├── data.py                # DataLoader, ToTensor, phân chia Train/Val/Test
│   │   ├── model.py               # Kiến trúc FashionMNISTMLP (Flatten -> Linear -> ReLU)
│   │   ├── train.py               # Script huấn luyện MLP và lưu best model
│   │   ├── evaluate.py            # Đánh giá độc lập trên 10,000 ảnh test
│   │   └── predict.py             # Dự đoán nhãn và xác suất cho mẫu ảnh đơn lẻ
│   └── cifar10/                   # Bài toán B: Phân loại vật thể màu với CNN
│       ├── __init__.py
│       ├── data.py                # DataLoader, Data Augmentation tách biệt, mirror
│       ├── model.py               # Kiến trúc CIFAR10CNN (Conv2d -> ReLU -> MaxPool)
│       ├── train.py               # Script huấn luyện CNN với các tùy chọn kênh/augment
│       ├── evaluate.py            # Đánh giá độc lập trên tập test CIFAR-10
│       └── predict.py             # Dự đoán và xuất Top-3 nhãn có xác suất cao nhất
├── tests/
│   ├── __init__.py
│   ├── test_models.py             # Kiểm tra shape, finite values, forward/backward
│   └── test_checkpoint.py         # Kiểm tra lưu/nạp state_dict và tính bảo toàn trọng số
├── data/                          # Thư mục chứa dữ liệu thô (được .gitignore bỏ qua)
├── checkpoints/                   # Thư mục lưu file .pt (được .gitignore bỏ qua)
└── reports/
    └── experiments.md             # Báo cáo các thực nghiệm kiểm soát siêu tham số
```

---

## 4. Cài đặt và Xác minh Môi trường

### 4.1. Cài đặt Dependencies

Dự án chỉ yêu cầu `torch`, `torchvision`, và `pytest`.

```bash
# Cài đặt phiên bản tiêu chuẩn (mặc định CPU trên Windows)
pip install -r requirements.txt
```

> **Lưu ý về GPU (CUDA):**
> Nếu máy tính của bạn có card màn hình NVIDIA và bạn muốn tận dụng CUDA, hãy truy cập công cụ chọn lệnh cài đặt chính thức của PyTorch:
> 👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
>
> Ví dụ lệnh cài PyTorch với CUDA 12.4:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 4.2. Chẩn đoán Môi trường (Stage 0)

Chạy module chẩn đoán từ thư mục gốc của repository:

```bash
python -m pytorch_lab.check_environment
```

**Kỳ vọng kết quả hiển thị:**
- Phiên bản Python, PyTorch, TorchVision.
- Trạng thái CUDA (Có/Không), tên GPU được phát hiện (nếu có).
- Thiết bị được hệ thống tự động chỉ định (`cpu` hoặc `cuda`).
- Kết quả phép nhân ma trận kiểm chứng chạy trực tiếp trên thiết bị đã chọn.

---

## 5. Stage 1 — Học Tensor và Autograd từ Gốc

Trước khi bắt đầu huấn luyện mô hình ảnh, hãy chạy tập lệnh tương tác này để nắm vững 8 khái niệm tính toán cốt lõi:

```bash
python -m pytorch_lab.tensors_tutorial
```

**Nội dung được minh họa chi tiết bằng tiếng Việt:**
1. **Khởi tạo Tensor**: `tensor`, `zeros`, `ones`, `arange`, `randn`.
2. **4 thuộc tính nền tảng**: `.shape`, `.ndim`, `.dtype`, `.device`.
3. **Cắt lát (Slicing)**: Cắt hàng, cột, và lọc theo điều kiện logic (Boolean Masking).
4. **Biến đổi hình dạng (Reshaping/Flattening)**: Tại sao ảnh 2D `[B, 1, 28, 28]` phải đổi thành `[B, 784]` trước khi đưa vào MLP.
5. **Nhân ma trận**: Phân biệt nhân từng phần tử `A * B` với nhân đại số tuyến tính `A @ B`.
6. **Autograd**: `requires_grad=True` và đồ thị tính toán (Computational Graph).
7. **Đạo hàm tự động**: Lan truyền ngược với hàm `backward()` và giải thích đạo hàm toán học.
8. **Bản chất Gradient Descent**: Công thức $W_{mới} = W_{cũ} - \eta \cdot \nabla W$ và lý do bắt buộc phải gọi `zero_grad()`.

---

## 6. Hướng dẫn Lệnh Thực thi (Train, Evaluate, Predict)

Tất cả các lệnh đều được chạy từ **thư mục gốc của repository** bằng cú pháp chuẩn của Python `python -m ...`:

### 6.1. Bài toán A — FashionMNIST (MLP)

```bash
# 1. Huấn luyện mô hình MLP (mặc định 5 epochs, lr=0.001)
python -m pytorch_lab.fashion_mnist.train --epochs 5 --lr 0.001 --batch-size 64

# 2. Đánh giá mô hình tốt nhất trên tập kiểm tra độc lập (10,000 ảnh)
python -m pytorch_lab.fashion_mnist.evaluate

# 3. Dự đoán nhãn cho một mẫu ảnh cụ thể (ví dụ ảnh số 0 hoặc số 12)
python -m pytorch_lab.fashion_mnist.predict --sample-idx 0
python -m pytorch_lab.fashion_mnist.predict --sample-idx 12
```

### 6.2. Bài toán B — CIFAR-10 (CNN)

```bash
# 1. Huấn luyện mô hình CNN baseline
python -m pytorch_lab.cifar10.train --epochs 5 --lr 0.001 --batch-size 64

# 2. Huấn luyện có bật tính năng tăng cường dữ liệu (Data Augmentation)
python -m pytorch_lab.cifar10.train --epochs 5 --lr 0.001 --augment

# 3. Đánh giá mô hình CNN trên 10,000 ảnh test CIFAR-10
python -m pytorch_lab.cifar10.evaluate

# 4. Dự đoán nhãn cho mẫu ảnh kiểm tra
python -m pytorch_lab.cifar10.predict --sample-idx 1
python -m pytorch_lab.cifar10.predict --sample-idx 3
```

### 6.3. Chạy Toàn bộ Kiểm thử Tự động (Unit Tests)

```bash
python -m pytest tests -v
```
*(Chạy hoàn toàn trên CPU, không yêu cầu tải dataset, kiểm tra đầy đủ shape, tính hữu hạn của giá trị, forward/backward smoke test, và lưu/nạp checkpoint).*

---

## 7. Hình dạng Dữ liệu (Shapes) và Bảng Nhãn Lớp

### 7.1. Bảng đối chiếu Kích thước Tensor

| Thuộc tính | FashionMNIST | CIFAR-10 |
| :--- | :--- | :--- |
| **Kích thước ảnh thô** | 28 x 28 pixels, 1 kênh xám | 32 x 32 pixels, 3 kênh màu (R, G, B) |
| **Tensor 1 ảnh** | `[1, 28, 28]` | `[3, 32, 32]` |
| **Tensor Batch (B=64)** | `[64, 1, 28, 28]` | `[64, 3, 32, 32]` |
| **Sau Flatten / Conv** | Trải phẳng thành `[64, 784]` | Qua 2 tầng Conv+Pool thành `[64, 4096]` |
| **Output Logits** | `[64, 10]` | `[64, 10]` |

### 7.2. Bảng Nhãn Lớp (Class Names)

| ID | Nhãn FashionMNIST | Nghĩa tiếng Việt | Nhãn CIFAR-10 | Nghĩa tiếng Việt |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `T-shirt/top` | Áo thun / áo cộc | `airplane` | Máy bay |
| **1** | `Trouser` | Quần dài | `automobile` | Ô tô con |
| **2** | `Pullover` | Áo len chui đầu | `bird` | Chim |
| **3** | `Dress` | Váy liền thân | `cat` | Mèo |
| **4** | `Coat` | Áo khoác | `deer` | Hươu / Nai |
| **5** | `Sandal` | Dép quai hậu | `dog` | Chó |
| **6** | `Shirt` | Áo sơ mi | `frog` | Ếch |
| **7** | `Sneaker` | Giày thể thao | `horse` | Ngựa |
| **8** | `Bag` | Túi xách | `ship` | Tàu thủy |
| **9** | `Ankle boot` | Ủng / Bốt cổ thấp | `truck` | Xe tải |

---

## 8. Phân tích Vòng lặp Huấn luyện (Step-by-Step)

Vòng lặp huấn luyện trong `pytorch_lab/common/engine.py` được thiết kế theo đúng quy chuẩn khoa học gồm 8 bước tuần tự:

```python
# 1. Kích hoạt chế độ huấn luyện
model.train()

for inputs, targets in dataloader:
    # 2. Đưa dữ liệu lên thiết bị (GPU hoặc CPU)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # 3. Xóa gradient tích lũy từ batch trước
    optimizer.zero_grad()

    # 4. Lan truyền xuôi: Dự đoán logits thô
    outputs = model(inputs)

    # 5. Tính hàm mất mát bằng CrossEntropyLoss
    loss = criterion(outputs, targets)

    # 6. Lan truyền ngược: Tính đạo hàm riêng cho từng tham số
    loss.backward()

    # 7. Cập nhật trọng số theo hướng giảm gradient
    optimizer.step()

    # 8. Cộng dồn số liệu thống kê chuẩn xác
    running_loss += loss.item() * targets.size(0)
    correct_samples += (outputs.argmax(dim=-1) == targets).sum().item()
```

### Tại sao mô hình trả về Logits thô mà KHÔNG dùng Softmax ở tầng cuối?
Trong PyTorch, `nn.CrossEntropyLoss()` thực hiện toán học tương đương với `LogSoftmax + NLLLoss` nhưng được tối ưu hóa trong một hàm nội bộ duy nhất. Kỹ thuật số học này sử dụng mẹo Log-Sum-Exp để tránh tràn số (numerical overflow) khi các giá trị lũy thừa $e^z$ quá lớn hoặc triệt tiêu về 0 (underflow) khi $e^z$ quá nhỏ. Nếu áp dụng `Softmax` bên trong mô hình rồi truyền vào `CrossEntropyLoss`, gradient sẽ bị sai lệch hoàn toàn. Khi cần xác suất phần trăm, ta chỉ gọi `torch.softmax(logits, dim=-1)` trong bước suy diễn `predict.py`.

---

## 9. Sự khác biệt giữa Validation và Test Evaluation

| Tiêu chí | Tập Huấn luyện (Train Set) | Tập Xác thực (Validation Set) | Tập Kiểm tra (Test Set) |
| :--- | :--- | :--- | :--- |
| **Mục đích** | Dùng để cập nhật trọng số mô hình qua gradient | Theo dõi hiện tượng quá khớp (overfitting) và lưu Best Checkpoint | Đánh giá năng lực tổng quát hóa khách quan cuối cùng |
| **Cập nhật tham số** | **CÓ** (`backward()` và `step()`) | **KHÔNG** (`torch.inference_mode()`) | **KHÔNG** (`torch.inference_mode()`) |
| **Tần suất gọi** | Mỗi epoch | Sau mỗi epoch | Chỉ gọi duy nhất **1 lần** sau khi đã chọn xong mô hình |
| **Sử dụng để chọn siêu tham số?** | Không | **CÓ** (chọn epoch tốt nhất, chọn lr, chọn số tầng) | **TUYỆT ĐỐI KHÔNG** |

> **Quy tắc vàng của Machine Learning:** Không bao giờ sử dụng tập Test để đưa ra bất kỳ quyết định tinh chỉnh tham số nào. Nếu dùng Test set để chọn mô hình, mô hình của bạn đã bị rò rỉ dữ liệu (data leakage) và độ chính xác báo cáo sẽ bị thổi phồng giả tạo.

---

## 10. Cách kiểm tra Checkpoint đã lưu

Các file checkpoint được lưu dưới định dạng `.pt` chuẩn tại thư mục `checkpoints/`. File này là một từ điển Python chứa:
- `model_state_dict`: Toàn bộ ma trận trọng số và bias của từng tầng.
- `optimizer_state_dict`: Trạng thái động lượng (momentum) của thuật toán tối ưu.
- `epoch`: Epoch đã lưu checkpoint.
- `val_acc` & `val_loss`: Kết quả kiểm định tại thời điểm lưu.
- `model_config`: Siêu dữ liệu kiến trúc (số kênh, số nơ-ron).
- `class_names`: Danh sách tên các nhãn lớp.

Bạn có thể kiểm tra nội dung checkpoint bằng đoạn mã ngắn:

```python
import torch

checkpoint = torch.load("checkpoints/fashion_mnist/best_model.pt", map_location="cpu", weights_only=False)
print("Các khóa trong checkpoint:", checkpoint.keys())
print("Epoch đã lưu:", checkpoint["epoch"])
print("Độ chính xác Validation:", checkpoint["val_acc"])
print("Cấu hình mô hình:", checkpoint["model_config"])
```

---

## 11. Các lỗi thường gặp và Cách khắc phục

1. **Lỗi `ModuleNotFoundError: No module named 'pytorch_lab'`**:
   - *Nguyên nhân*: Bạn đang chạy từ một thư mục con hoặc không dùng cờ `-m`.
   - *Khắc phục*: Luôn đứng tại thư mục gốc `image_classification_lab` và chạy `python -m pytorch_lab.check_environment`.
2. **Lỗi `UnicodeEncodeError: 'charmap' codec can't encode character...` trên Windows**:
   - *Nguyên nhân*: Terminal Windows cmd/powershell mặc định dùng bảng mã cp1252.
   - *Khắc phục*: Tất cả các script trong repo đã tích hợp sẵn cơ chế tự động chuyển `sys.stdout` sang `utf-8`. Bạn cũng có thể thiết lập biến môi trường `set PYTHONIOENCODING=utf-8`.
3. **Lỗi mạng khi tải CIFAR-10 từ server Toronto**:
   - *Nguyên nhân*: Server `cs.toronto.edu` thường xuyên bị bóp băng thông xuống ~60 kB/s.
   - *Khắc phục*: Module `pytorch_lab.cifar10.data` đã tích hợp sẵn cơ chế tải dự phòng tự động từ mirror Hugging Face với tốc độ cao (chỉ mất ~20 giây).
4. **Lỗi Out-of-Memory (OOM) khi dùng GPU**:
   - *Khắc phục*: Giảm `--batch-size` từ 128 hoặc 64 xuống 32 hoặc 16.

---

## 12. Giới hạn Thực nghiệm và Hướng phát triển

- **Hiện trạng Baseline**: Hai mô hình MLP và CNN trong lab được xây dựng thuần túy theo nguyên lý tối giản, không sử dụng `BatchNorm` hay `Dropout` để học viên quan sát dòng chảy gradient rõ ràng nhất.
- **Hiện tượng Overfitting**: Ở các mô hình phức tạp hơn, khi số epoch tăng lên, Train Loss sẽ tiếp tục giảm nhưng Validation Loss có thể tăng ngược trở lại.
- **Các bước học tiếp theo (Next Steps)**:
  1. Thêm `nn.Dropout(p=0.5)` vào các tầng Fully Connected để chống hiện tượng phụ thuộc quá mức vào một số nơ-ron.
  2. Thêm `nn.BatchNorm2d` sau mỗi tầng `Conv2d` để ổn định phân phối dữ liệu trung gian và đẩy tốc độ học lên cao hơn.
  3. Thử nghiệm bộ lập lịch tốc độ học (`lr_scheduler`, ví dụ `StepLR` hoặc `CosineAnnealingLR`).
  4. Mở rộng sang các kiến trúc kinh điển như LeNet-5, AlexNet, hoặc Residual Connections (ResNet).
