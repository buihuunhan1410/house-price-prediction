import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Dự Đoán Giá Nhà California", layout="centered")

@st.cache_resource
def load_model():
    # Phải trùng tên với file trên GitHub của bạn
    with open("mo_hinh_random_forest.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🏠 Dự Đoán Giá Nhà California")

col1, col2 = st.columns(2)

with col1:
    thu_nhap = st.number_input("Thu nhập trung bình", value=3.5)
    tuoi_nha = st.number_input("Tuổi nhà trung bình", value=20)
    dan_so = st.number_input("Dân số khu vực", value=1000)
    so_ho = st.number_input("Số hộ gia đình", value=400)

with col2:
    kinh_do = st.number_input("Kinh độ", value=-122.2)
    vi_do = st.number_input("Vĩ độ", value=37.8)
    tong_phong = st.number_input("Tổng số phòng", value=1500)
    tong_ngu = st.number_input("Tổng số phòng ngủ", value=300)

vi_tri = st.selectbox("Vị trí so với biển", 
                     ['gan_vinh', 'trong_dat_lien', 'gan_bien', 'dao', '<1H OCEAN'])

if st.button("Dự Đoán Ngay"):
    try:
        # 1. Tạo DataFrame gốc (9 cột bạn nhập)
        input_data = pd.DataFrame([[
            kinh_do, vi_do, tuoi_nha, tong_phong, tong_ngu, dan_so, so_ho, thu_nhap, vi_tri
        ]], columns=['kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                     'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh', 'vi_tri_gan_bien'])

        # 2. TỰ TÍNH 3 CỘT CÒN THIẾU (Để đủ 12 cột như mô hình yêu cầu)
        input_data["phong_tren_moi_ho"] = input_data["tong_so_phong"] / input_data["so_ho_gia_dinh"]
        input_data["ty_le_phong_ngu"] = input_data["tong_so_phong_ngu"] / input_data["tong_so_phong"]
        input_data["dan_so_tren_moi_ho"] = input_data["dan_so"] / input_data["so_ho_gia_dinh"]

        # 3. Sắp xếp lại đúng thứ tự 12 cột mà Pipeline của bạn đã học
        columns_order = [
            'kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
            'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh',
            'phong_tren_moi_ho', 'ty_le_phong_ngu', 'dan_so_tren_moi_ho', 'vi_tri_gan_bien'
        ]
        input_final = input_data[columns_order]

        # Dự đoán
        prediction = model.predict(input_final)
        st.success(f"### 💰 Giá nhà dự báo: ${prediction[0]:,.2f}")
        
    except Exception as e:
        st.error(f"Lỗi: {e}")
