import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Cấu hình trang
st.set_page_config(page_title="Dự Đoán Giá Nhà California", layout="centered")

# 2. Load mô hình đã train
@st.cache_resource
def load_model():
    # Đảm bảo file .pkl nằm cùng thư mục với file app.py này
    with open("mo_hinh_random_forest_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# 3. Giao diện người dùng
st.title("🏠 Dự Đoán Giá Nhà California")
st.write("Nhập thông số căn nhà để nhận giá dự báo (đơn vị: USD)")

col1, col2 = st.columns(2)

with col1:
    thu_nhap = st.number_input("Thu nhập trung bình (chục ngàn USD)", value=3.5, step=0.1)
    tuoi_nha = st.number_input("Tuổi nhà trung bình", value=20, step=1)
    dan_so = st.number_input("Dân số khu vực", value=1000, step=100)
    so_ho = st.number_input("Số hộ gia đình", value=400, step=50)

with col2:
    kinh_do = st.number_input("Kinh độ (Longitude)", value=-122.2, format="%.2f")
    vi_do = st.number_input("Vĩ độ (Latitude)", value=37.8, format="%.2f")
    tong_phong = st.number_input("Tổng số phòng", value=1500, step=100)
    tong_ngu = st.number_input("Tổng số phòng ngủ", value=300, step=50)

vi_tri = st.selectbox("Vị trí so với biển", 
                     ['<1H OCEAN', 'trong_dat_lien', 'gan_vinh', 'gan_bien', 'dao'])

# 4. Xử lý dự đoán
if st.button("Dự Đoán Ngay"):
    # Tạo DataFrame từ input
    input_data = pd.DataFrame([[
        kinh_do, vi_do, tuoi_nha, tong_phong, tong_ngu, dan_so, so_ho, thu_nhap, vi_tri
    ]], columns=['kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                 'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh', 'vi_tri_gan_bien'])

    # Dự đoán
    try:
        prediction = model.predict(input_data)
        st.success(f"### 💰 Giá nhà dự báo: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Lỗi: {e}. Hãy đảm bảo mô hình .pkl đã bao gồm các bước tiền xử lý (Pipeline).")
