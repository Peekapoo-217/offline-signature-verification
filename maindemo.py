import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import pickle  # Dùng để lưu database vĩnh viễn
import os
from collections import OrderedDict

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & CORE DATABASE
# ==========================================
st.set_page_config(page_title="Hệ thống Kiểm định Chữ Ký Số - AI", page_icon="🛡️", layout="wide")

st.title("🛡️ Hệ thống Giám định & Xác thực Chữ ký Tự động")
st.markdown("### *Ứng dụng lõi Siamese Network - Phân tích đặc trưng hình học Zero-shot*")
st.divider()

MODEL_PATH = "best_model.pth"
DB_PATH = "signature_database.pkl"  # File lưu trữ vector chữ ký mẫu
THRESHOLD = 0.3494
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tự động tải/khởi tạo Database vĩnh viễn
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        st.session_state['database'] = pickle.load(f)
else:
    st.session_state['database'] = {}

# ==========================================
# 2. MÔ HÌNH VÀ TIỀN XỬ LÝ (GIỮ NGUYÊN CORE XỊN)
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=128):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, embed_dim)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return F.normalize(x, p=2, dim=1)

@st.cache_resource 
def load_model():
    if not os.path.exists(MODEL_PATH): return None
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

IMG_HEIGHT, IMG_WIDTH = 128, 256

def preprocess_signature_matched(image_gray):
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        thresh = thresh[y:y+h, x:x+w]
    else:
        h, w = thresh.shape
    scale = min(IMG_HEIGHT / h, IMG_WIDTH / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_w == 0 or new_h == 0:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    y_offset = (IMG_HEIGHT - new_h) // 2
    x_offset = (IMG_WIDTH - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return preprocess_signature_matched(gray), img

def image_to_tensor(processed_img):
    return torch.FloatTensor(processed_img / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

model = load_model()
if model is None:
    st.error(f"❌ Không tìm thấy file trọng số `{MODEL_PATH}` ở thư mục hiện tại!")
    st.stop()

# ==========================================
# 3. GIAO DIỆN PHÂN HỆ CHỨC NĂNG (UX THỰC TẾ)
# ==========================================
st.sidebar.title("💳 Phân hệ quản lý")
page = st.sidebar.radio("Chọn nghiệp vụ:", ["📂 Khởi tạo Hồ sơ Gốc", "🔍 Giám định chữ ký hồ sơ"])
st.sidebar.divider()
st.sidebar.metric(label="Tổng số hồ sơ trong DB", value=len(st.session_state['database']))

# 📂 NGHIỆP VỤ 1: KHỞI TẠO HỒ SƠ GỐC (ENROLL)
if page == "📂 Khởi tạo Hồ sơ Gốc":
    st.header("📂 Khởi tạo dữ liệu Chữ ký Gốc (Hồ sơ Khách hàng)")
    st.caption("Thiết lập mẫu chữ ký chuẩn từ tài liệu tùy thân (CCCD/Hộ chiếu) phục vụ đối chiếu sau này.")
    
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        user_name = st.text_input("Mã định danh khách hàng (Họ và tên / Số CMND):")
        uploaded = st.file_uploader("Tải lên ảnh chữ ký trích xuất từ văn bản gốc:", type=["jpg", "png", "jpeg"])
        
        if st.button("Xác nhận Lưu trữ Hồ sơ", type="primary"):
            if not user_name:
                st.warning("⚠️ Vui lòng điền mã định danh khách hàng!")
            elif uploaded is None:
                st.warning("⚠️ Vui lòng tải lên tài liệu chữ ký!")
            else:
                proc_img, _ = preprocess_uploaded_image(uploaded)
                if proc_img is not None:
                    tensor_img = image_to_tensor(proc_img)
                    with torch.no_grad():
                        vector = model.forward_once(tensor_img).cpu() # Chuyển về CPU để lưu pickle
                    
                    # Lưu trữ cặp dữ liệu bao gồm cả ảnh đã xử lý để hiển thị lại
                    st.session_state['database'][user_name] = {
                        'vector': vector,
                        'processed_img': proc_img
                    }
                    # Ghi file lưu lại
                    with open(DB_PATH, 'wb') as f:
                        pickle.dump(st.session_state['database'], f)
                        
                    st.success(f"🎉 Đã đồng bộ và lưu trữ vĩnh viễn cấu trúc chữ ký của: **{user_name}**")
                    st.rerun()

    with col_preview:
        if uploaded is not None:
            proc_img, orig_img = preprocess_uploaded_image(uploaded)
            st.image(orig_img, caption="Ảnh tài liệu gốc", channels="BGR", width=250)
            st.image(proc_img, caption="AI Trích xuất cạnh hình học (128x256)", width=250)

# 🔍 NGHIỆP VỤ 2: GIÁM ĐỊNH CHỮ KÝ HỒ SƠ (VERIFY)
elif page == "🔍 Giám định chữ ký hồ sơ":
    st.header("🔍 Giám định số thực tài liệu trực tuyến")
    st.caption("Tải lên một tài liệu, hợp đồng mới cần kiểm thử tính pháp lý của chữ ký.")

    if not st.session_state['database']:
        st.info("💡 Hệ thống hiện đang rỗng. Vui lòng quay lại tab 'Khởi tạo Hồ sơ Gốc' để nhập dữ liệu mẫu trước.")
    else:
        selected_user = st.selectbox("Hồ sơ đối chiếu (Khách hàng mục tiêu):", list(st.session_state['database'].keys()))
        uploaded_v = st.file_uploader("Tải lên ảnh chữ ký lấy từ văn bản cần thẩm định:", type=["jpg", "png", "jpeg"])

        if uploaded_v is not None:
            proc_v, orig_v = preprocess_uploaded_image(uploaded_v)
            
            if st.button("Tiến hành Giám định AI", type="primary"):
                # Lấy dữ liệu hồ sơ gốc
                db_data = st.session_state['database'][selected_user]
                vector_old = db_data['vector'].to(DEVICE)
                proc_old = db_data['processed_img']
                
                # Trích xuất dữ liệu chữ ký kiểm tra
                tensor_v = image_to_tensor(proc_v)
                with torch.no_grad():
                    vector_new = model.forward_once(tensor_v)
                    dist = F.pairwise_distance(vector_old, vector_new).item()
                
                # Phán quyết hệ thống
                is_valid = dist < THRESHOLD
                
                st.markdown("---")
                st.subheader("📊 KẾT QUẢ PHÂN TÍCH TRỰC QUAN")
                
                # Giao diện hiển thị song song 2 ảnh đối chiếu cực kỳ chuyên nghiệp
                col_side1, col_side2 = st.columns(2)
                with col_side1:
                    st.image(proc_old, caption=f"Mẫu chuẩn lưu trong hồ sơ của {selected_user}", width=300)
                with col_side2:
                    st.image(proc_v, caption="Mẫu trích xuất trên tài liệu mới cần kiểm tra", width=300)
                
                st.divider()
                
                # Hiển thị kết luận dạng DashBoard tài chính/ngân hàng
                col_res1, col_res2 = st.columns([2, 1])
                with col_res1:
                    if is_valid:
                        st.success(f"### Kết luận: CHỮ KÝ HỢP LỆ\n\nMô hình AI xác nhận chữ ký này trùng khớp với chữ ký gốc của khách hàng **{selected_user}**.")
                    else:
                        st.error(f"### Kết luận: PHÁT HIỆN GIẢ MẠO / SAI LỆCH\n\nKhoảng cách đặc trưng vượt ngưỡng cho phép. Chữ ký này KHÔNG PHẢI do chủ hồ sơ **{selected_user}** thực hiện.")
                
                with col_res2:
                    st.metric(
                        label="Khoảng cách sai lệch hình học (Distance)", 
                        value=f"{dist:.4f}", 
                        delta=f"Ngưỡng an toàn: {THRESHOLD}",
                        delta_color="inverse" if not is_valid else "normal"
                    )