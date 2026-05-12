import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from collections import OrderedDict
import os

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hệ thống Xác thực Chữ Ký AI", page_icon="✍️", layout="centered")
st.title("✍️ Xác thực Chữ ký bằng Trí tuệ Nhân tạo")
st.markdown("Đồ án: Mô hình Siamese Network (ResNet18) - Zero-shot Verification")

# Đường dẫn đến file model và Ngưỡng xác thực
MODEL_PATH = "best_model.pth"
THRESHOLD = 0.3158
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo Database giả lập trên RAM (Session State)
if 'database' not in st.session_state:
    st.session_state['database'] = {}

# ==========================================
# 2. KIẾN TRÚC MODEL & TIỀN XỬ LÝ
# ==========================================
class SiameseNetwork(nn.Module):
    """Kiến trúc KHỚP CHÍNH XÁC với model.py lúc train."""
    def __init__(self, embed_dim=128):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),      # Khớp với model.py (0.5, KHÔNG phải 0.6)
            nn.Linear(512, embed_dim)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x                    # KHÔNG normalize — khớp với lúc train

@st.cache_resource # Cache để không phải load model nhiều lần gây nặng web
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = SiameseNetwork(embed_dim=128).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# --- Tiền xử lý KHỚP CHÍNH XÁC với preprocessing.py lúc train ---
IMG_SIZE = 128  # Ảnh vuông 128x128 giống lúc train

def preprocess_signature_matched(image_gray):
    """
    Pipeline GIỐNG HỆT preprocessing.py:
      Grayscale -> Adaptive Threshold (Gaussian, blockSize=11, C=2) -> Resize 128x128
    """
    thresh = cv2.adaptiveThreshold(
        image_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_canvas_image(img_array):
    """Chuyển ảnh RGBA từ Canvas web -> pipeline chuẩn lúc train."""
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    return preprocess_signature_matched(gray)

def preprocess_uploaded_image(uploaded_file):
    """Chuyển ảnh upload (JPG/PNG) -> pipeline chuẩn lúc train."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return preprocess_signature_matched(gray)

def image_to_tensor(processed_img):
    """Ảnh đã xử lý (128x128) -> Tensor sẵn sàng đưa vào model."""
    return torch.FloatTensor(processed_img / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ==========================================
# 3. GIAO DIỆN WEB DEMO
# ==========================================
model = load_model()

if model is None:
    st.error(f"❌ Không tìm thấy file `{MODEL_PATH}`. Hãy copy file model vào cùng thư mục với `app.py`!")
    st.stop()

# Dùng Sidebar menu thay vì Tabs (tránh bug canvas không render ở tab thứ 2)
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("Chọn chức năng:", ["📝 Đăng ký chữ ký", "🔍 Xác thực chữ ký"])
st.sidebar.divider()
st.sidebar.markdown(f"**Người dùng đã đăng ký:** {len(st.session_state['database'])}")
if st.session_state['database']:
    for name in st.session_state['database']:
        st.sidebar.write(f"• {name}")

# ==================== TRANG ĐĂNG KÝ ====================
if page == "📝 Đăng ký chữ ký":
    st.header("Bước 1: Đăng ký chữ ký mẫu")
    user_id = st.text_input("Nhập tên của bạn (VD: nguyen_van_a):", key="enroll_id")

    enroll_mode = st.radio("Chọn cách nhập chữ ký:", ["Vẽ tay trên web", "Upload ảnh chữ ký"], key="enroll_mode", horizontal=True)

    processed_img = None
    if enroll_mode == "Vẽ tay trên web":
        st.write("Dùng chuột (hoặc ngón tay) ký vào khung dưới đây:")
        canvas_result = st_canvas(
            fill_color="white", stroke_width=3, stroke_color="black",
            background_color="white", height=200, width=600,
            drawing_mode="freedraw", key="canvas_enroll",
        )
        if canvas_result.image_data is not None:
            processed_img = preprocess_canvas_image(canvas_result.image_data)
    else:
        uploaded = st.file_uploader("Tải lên ảnh chữ ký (JPG/PNG):", type=["jpg", "jpeg", "png"], key="enroll_upload")
        if uploaded is not None:
            processed_img = preprocess_uploaded_image(uploaded)
            if processed_img is None:
                st.error("Không đọc được ảnh. Vui lòng thử file khác!")

    if st.button("Lưu Chữ Ký (Đăng ký)", type="primary"):
        if not user_id:
            st.warning("Vui lòng nhập tên của bạn!")
        elif processed_img is not None:
            tensor_img = image_to_tensor(processed_img)
            with torch.no_grad():
                vector = model.forward_once(tensor_img)
            st.session_state['database'][user_id] = vector
            st.success(f"✅ Đã đăng ký thành công chữ ký cho: **{user_id}**")
            st.image(processed_img, caption="Ảnh model nhìn thấy (128×128, Adaptive Threshold)", width=256)
        else:
            st.warning("Vui lòng vẽ hoặc upload ảnh chữ ký trước!")

# ==================== TRANG XÁC THỰC ====================
elif page == "🔍 Xác thực chữ ký":
    st.header("Bước 2: Kiểm tra chữ ký")

    if not st.session_state['database']:
        st.info("Chưa có ai đăng ký. Vui lòng chọn 'Đăng ký chữ ký' ở menu bên trái trước!")
    else:
        user_list = list(st.session_state['database'].keys())
        selected_user = st.selectbox("Chọn người dùng để xác thực:", user_list)

        verify_mode = st.radio("Chọn cách nhập:", ["Vẽ tay trên web", "Upload ảnh chữ ký"], key="verify_mode", horizontal=True)

        processed_img_v = None
        if verify_mode == "Vẽ tay trên web":
            st.write(f"Vẽ lại chữ ký của **{selected_user}** vào khung dưới đây:")
            canvas_verify = st_canvas(
                fill_color="white", stroke_width=3, stroke_color="black",
                background_color="white", height=200, width=600,
                drawing_mode="freedraw", key="canvas_verify",
            )
            if canvas_verify.image_data is not None:
                processed_img_v = preprocess_canvas_image(canvas_verify.image_data)
        else:
            uploaded_v = st.file_uploader("Tải lên ảnh chữ ký cần kiểm tra:", type=["jpg", "jpeg", "png"], key="verify_upload")
            if uploaded_v is not None:
                processed_img_v = preprocess_uploaded_image(uploaded_v)
                if processed_img_v is None:
                    st.error("Không đọc được ảnh!")

        if st.button("Kiểm tra ngay", type="primary"):
            if processed_img_v is not None:
                tensor_img_v = image_to_tensor(processed_img_v)
                vector_old = st.session_state['database'][selected_user]

                with torch.no_grad():
                    vector_new = model.forward_once(tensor_img_v)
                    dist = F.pairwise_distance(vector_old, vector_new).item()

                st.divider()
                st.subheader("KẾT QUẢ ĐÁNH GIÁ:")
                if dist < THRESHOLD:
                    st.success(f"✅ ĐĂNG NHẬP THÀNH CÔNG! (Đúng là chữ ký của {selected_user})")
                    st.metric(label="Khoảng cách (Distance)", value=f"{dist:.4f}", delta="Hợp lệ", delta_color="normal")
                else:
                    st.error("❌ CHỮ KÝ GIẢ MẠO HOẶC KHÔNG KHỚP!")
                    st.metric(label="Khoảng cách (Distance)", value=f"{dist:.4f}", delta="Vượt ngưỡng", delta_color="inverse")

                st.image(processed_img_v, caption="Ảnh model nhìn thấy (128×128)", width=256)
            else:
                st.warning("Vui lòng vẽ hoặc upload ảnh chữ ký trước!")