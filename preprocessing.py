# ===========================================================================
# PREPROCESSING — Tiền xử lý ảnh chữ ký
# ===========================================================================
# Hàm này phải KHỚP CHÍNH XÁC với hàm preprocess_signature() đã dùng khi
# training để đảm bảo kết quả inference khớp với kết quả đánh giá.
# ===========================================================================

import numpy as np
import cv2
import torch

from config import IMG_SIZE


def preprocess_signature(
    image: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Tiền xử lý ảnh chữ ký.

    Steps:
      1. Chuyển sang grayscale (nếu cần).
      2. Adaptive Threshold: khử nhiễu, làm rõ nét bút.
      3. Resize về (img_size, img_size).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào (BGR hoặc Grayscale), đã decode từ bytes.
    img_size : int
        Kích thước đầu ra (vuông).

    Returns
    -------
    np.ndarray
        Ảnh grayscale đã tiền xử lý, shape (img_size, img_size), dtype uint8.
    """
    if image is None:
        return np.zeros((img_size, img_size), dtype=np.uint8)

    # Nếu ảnh có 3 kênh (BGR) -> chuyển grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Adaptive Threshold — giống lúc train
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )

    resized = cv2.resize(thresh, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized


def image_to_tensor(preprocessed: np.ndarray) -> torch.Tensor:
    """
    Chuyển ảnh đã tiền xử lý thành tensor sẵn sàng đưa vào model.

    Giống logic của Dataset khi KHÔNG có augmentation (Val/Test):
      img = torch.FloatTensor(img_raw / 255.0).unsqueeze(0)

    Output shape: (1, 1, IMG_SIZE, IMG_SIZE)  — batch_size=1, channels=1
    """
    tensor = torch.FloatTensor(preprocessed / 255.0).unsqueeze(0)  # (1, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 1, H, W) — thêm batch dim
    return tensor
