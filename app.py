import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

# 1. Cấu hình giao diện
st.set_page_config(page_title="Dự Đoán Giá Nhà California", layout="wide")

# 2. Hàm Load mô hình (Xử lý lỗi AttributeError và phiên bản)
@st.cache_resource
def load_model():
    try:
        # Đảm bảo file mo_hinh_random_forest.pkl nằm cùng thư mục với app.py
        with open("mo_hinh_random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return None

model = load_model()

# 3. Giao diện người dùng
st.title("🏠 Ứng Dụng Dự Đoán Giá Nhà California")
st.info("Nhập các thông số cơ bản, hệ thống sẽ tự động tính toán các chỉ số nâng cao cho mô hình.")

if model is not None:
    # Chia làm 2 cột để nhập liệu cho đẹp
    col1, col2 = st.columns(2)

    with col1:
        thu_nhap = st.number_input("Thu nhập trung bình (Median Income)", value=3.5)
        tuoi_nha = st.number_input("Tuổi nhà trung bình (Housing Median Age)", value=20)
        dan_so = st.number_input("Dân số khu vực (Population)", value=1000)
        so_ho = st.number_input("Số hộ gia đình (Households)", value=400)

    with col2:
        kinh_do = st.number_input("Kinh độ (Longitude)", value=-122.2)
        vi_do = st.number_input("Vĩ độ (Latitude)", value=37.8)
        tong_phong = st.number_input("Tổng số phòng (Total Rooms)", value=1500)
        tong_ngu = st.number_input("Tổng số phòng ngủ (Total Bedrooms)", value=300)

    vi_tri = st.selectbox("Vị trí so với biển (Ocean Proximity)", 
                         ['gan_vinh', 'trong_dat_lien', 'gan_bien', 'dao', '<1H OCEAN'])

    # 4. Nút dự đoán và xử lý dữ liệu
    if st.button("Dự Đoán Giá Nhà"):
        try:
            # Bước A: Tạo DataFrame từ 9 cột đầu vào
            input_df = pd.DataFrame([[
                kinh_do, vi_do, tuoi_nha, tong_phong, tong_ngu, dan_so, so_ho, thu_nhap, vi_tri
            ]], columns=['kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                         'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh', 'vi_tri_gan_bien'])

            # Bước B: TỰ TÍNH 3 CỘT PHÁI SINH (Vì file pkl yêu cầu đủ 12 cột)
            # Pipeline của bạn cần 11 cột số và 1 cột chữ = 12 cột
            input_df["phong_tren_moi_ho"] = input_df["tong_so_phong"] / input_df["so_ho_gia_dinh"]
            input_df["ty_le_phong_ngu"] = input_df["tong_so_phong_ngu"] / input_df["tong_so_phong"]
            input_df["dan_so_tren_moi_ho"] = input_df["dan_so"] / input_df["so_ho_gia_dinh"]

            # Bước C: Sắp xếp lại đúng thứ tự cột mà mô hình đã học trong Colab
            columns_order = [
                'kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh',
                'phong_tren_moi_ho', 'ty_le_phong_ngu', 'dan_so_tren_moi_ho', 'vi_tri_gan_bien'
            ]
            input_final = input_df[columns_order]

            # Bước D: Thực hiện dự đoán
            prediction = model.predict(input_final)
            
            # Hiển thị kết quả
            st.success(f"### 💰 Giá nhà dự báo: ${prediction[0]:,.2f}")
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
            st.warning("Gợi ý: Hãy đảm bảo phiên bản scikit-learn trên Streamlit giống với lúc train mô hình.")

else:
    st.warning("⚠️ Không thể chạy ứng dụng vì lỗi nạp mô hình.")
