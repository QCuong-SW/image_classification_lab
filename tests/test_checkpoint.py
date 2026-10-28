"""
Kiểm thử lưu và nạp checkpoint (Checkpoint Management Tests).

Các bài test kiểm chứng:
1. Lưu `state_dict` và nạp lại bảo toàn nguyên vẹn giá trị dự đoán (Prediction preservation).
2. Kiểm tra `torch.allclose` giữa kết quả suy diễn của mô hình ban đầu và mô hình sau khi nạp.
3. Kiểm tra các siêu dữ liệu đính kèm (metadata: val_acc, epoch, config, class_names).
4. Kiểm tra ném ngoại lệ `FileNotFoundError` khi đường dẫn checkpoint không tồn tại.
"""

import os
import pytest
import torch

from pytorch_lab.fashion_mnist.model import FashionMNISTMLP
from pytorch_lab.cifar10.model import CIFAR10CNN
from pytorch_lab.common.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load_mlp(tmp_path) -> None:
    """Kiểm tra lưu và nạp checkpoint cho mô hình MLP."""
    ckpt_file = str(tmp_path / "test_mlp_checkpoint.pt")

    # 1. Khởi tạo mô hình và lấy dự đoán ban đầu
    model_orig = FashionMNISTMLP(hidden_dim1=64, hidden_dim2=32)
    model_orig.eval()

    dummy_input = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        orig_preds = model_orig(dummy_input)

    # 2. Lưu checkpoint
    save_checkpoint(
        filepath=ckpt_file,
        model=model_orig,
        epoch=3,
        val_loss=0.45,
        val_acc=0.85,
        model_config=model_orig.get_config(),
        class_names=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
    )

    assert os.path.exists(ckpt_file), "File checkpoint chưa được tạo thành công!"

    # 3. Khởi tạo mô hình mới và nạp checkpoint
    model_loaded = FashionMNISTMLP(hidden_dim1=64, hidden_dim2=32)
    meta = load_checkpoint(filepath=ckpt_file, model=model_loaded)

    # 4. Kiểm tra siêu dữ liệu
    assert meta["epoch"] == 3
    assert meta["val_acc"] == 0.85
    assert len(meta["class_names"]) == 10

    # 5. Kiểm tra kết quả dự đoán khớp nhau tuyệt đối (numerical tolerance 1e-6)
    with torch.no_grad():
        loaded_preds = model_loaded(dummy_input)

    assert torch.allclose(orig_preds, loaded_preds, atol=1e-6), (
        "Dự đoán trước và sau khi nạp checkpoint không khớp nhau!"
    )


def test_checkpoint_save_and_load_cnn(tmp_path) -> None:
    """Kiểm tra lưu và nạp checkpoint cho mô hình CNN."""
    ckpt_file = str(tmp_path / "test_cnn_checkpoint.pt")

    model_orig = CIFAR10CNN(conv1_channels=16, conv2_channels=32, fc_hidden_dim=64)
    model_orig.eval()

    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        orig_preds = model_orig(dummy_input)

    save_checkpoint(
        filepath=ckpt_file,
        model=model_orig,
        val_acc=0.72,
    )

    model_loaded = CIFAR10CNN(conv1_channels=16, conv2_channels=32, fc_hidden_dim=64)
    load_checkpoint(filepath=ckpt_file, model=model_loaded)

    with torch.no_grad():
        loaded_preds = model_loaded(dummy_input)

    assert torch.allclose(orig_preds, loaded_preds, atol=1e-6)


def test_checkpoint_missing_file() -> None:
    """Kiểm tra ngoại lệ khi đường dẫn checkpoint không tồn tại."""
    model = FashionMNISTMLP()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(filepath="duong_dan_khong_ton_tai/model.pt", model=model)
