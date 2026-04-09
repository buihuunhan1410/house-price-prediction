import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

# 1. Cấu hình trang
st.set_page_config(page_title="Dự Đoán Giá Nhà California", layout="centered")

# 2. Hàm load mô hình
@st.cache_resource
def load_model():
    try:
        # File phải tên là mo_hinh_random_forest.pkl trên GitHub
        with open("mo_hinh_random_forest.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi load mô hình: {e}")
        return None

model = load_model()

# 3. Giao diện
st.title("🏠 Dự Đoán Giá Nhà California")
st.write("Vui lòng nhập thông số căn nhà:")

if model:
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
            # Tạo DataFrame từ 9 cột nhập vào
            data = pd.DataFrame([[
                kinh_do, vi_do, tuoi_nha, tong_phong, tong_ngu, dan_so, so_ho, thu_nhap, vi_tri
            ]], columns=['kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                         'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh', 'vi_tri_gan_bien'])

            # TỰ TÍNH 3 CỘT PHÁI SINH ĐỂ ĐỦ 12 CỘT
            data["phong_tren_moi_ho"] = data["tong_so_phong"] / data["so_ho_gia_dinh"]
            data["ty_le_phong_ngu"] = data["tong_so_phong_ngu"] / data["tong_so_phong"]
            data["dan_so_tren_moi_ho"] = data["dan_so"] / data["so_ho_gia_dinh"]

            # Sắp xếp đúng thứ tự mà Pipeline yêu cầu
            order = ['kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                     'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh',
                     'phong_tren_moi_ho', 'ty_le_phong_ngu', 'dan_so_tren_moi_ho', 'vi_tri_gan_bien']
            
            final_df = data[order]
            
            # Dự đoán
            ket_qua = model.predict(final_df)
            st.success(f"### 💰 Giá nhà dự báo: ${ket_qua[0]:,.2f}")
            
        except Exception as e:
            st.error(f"Lỗi tính toán: {e}")
