import streamlit as st
from web_function import predict

def app(df, x, y):
    st.markdown("<h1 style='text-align: center;'>Prediksi Tsunami</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        date_time = st.slider("Tentukan nilai bulan", key="date", min_value=1, max_value=12, value=1)
        magnitude = st.text_input("Masukkan nilai magnitude", key="mag")
        depth = st.text_input("Masukkan nilai depth", key="depth")
    with col2:
        alert = st.text_input("Masukkan nilai alert", key="alert")
        latitude = st.text_input("Masukkan nilai latitude", key="lat")
        longitude = st.text_input("Masukkan nilai longitude", key="long")

    # Konversi nilai magnitude menjadi string
    magnitude_str = str(magnitude)

    # Menggunakan metode strip() pada variabel features
    features = [magnitude_str.strip(), depth.strip(), str(date_time), alert.strip(), latitude.strip(), longitude.strip()]

    if st.button("Prediksi"):
        if any(value.strip() == '' for value in features):
            st.warning("Input tidak boleh kosong. Harap isi semua nilai.")
        else:
            prediction, score = predict(x, y, features)
            st.info("Prediksi sukses")

            if prediction == 1:
                st.warning("Data ini menyebabkan Terjadi Tsunami")
            else:
                st.success("Data ini TIDAK menyebabkan Tsunami")

            if score is not None:
                st.write("Model yang digunakan memiliki tingkat akurasi: {:.2f}%".format(score * 100))
