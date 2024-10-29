import streamlit as st
from web_function import predict

def app(df, x, y):
    st.markdown("<h1 style='text-align: center;'>Klasifikasi Tsunami</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    # Mengambil input dari pengguna, semua data sudah numerik
    with col1:
        magnitude = st.number_input("magnitude", min_value=0.0, value=7.0, step=0.1)
        cdi = st.number_input("cdi", min_value=0, value=8)
        mmi = st.number_input("mmi", min_value=0, value=5) 
        alert = st.number_input("alert", min_value=0, value=0)
        sig = st.number_input("sig", min_value=0, value=768)
        
        
    with col2:
        net = st.number_input("net", min_value=0, value=9)
        nst = st.number_input("nst", min_value=0, value=117)
        dmin = st.number_input("dmin", min_value=0.0, value=0.509, step=0.001)
        gap = st.number_input("gap", min_value=0.0, value=17.0, step=0.1)
        

    with col3:
        magType = st.number_input("magType", min_value=0, value=8)
        depth = st.number_input("depth", min_value=0.0, value=14.0, step=0.1)
        latitude = st.number_input("latitude", min_value=-90.0, max_value=90.0, value=-9.7963, step=0.0001)
        longitude = st.number_input("longitude", min_value=-180.0, max_value=180.0, value=159.596, step=0.0001)

    # Menggabungkan input dari pengguna sebagai fitur untuk prediksi (13 fitur termasuk mmi)
    features = [magnitude, cdi, mmi, alert, sig, net, nst, dmin, gap, magType, depth, latitude, longitude]

    if st.button("Prediksi"):
        # Prediksi dengan fitur yang sudah siap (semua numerik)
        prediction, score = predict(x, y, features)
        
        st.info("Prediksi sukses")
        
        # Tampilkan hasil prediksi
        if prediction == 1:
            st.warning("Data ini menyebabkan Terjadi Tsunami")
        else:
            st.success("Data ini TIDAK menyebabkan Tsunami")

        # Tampilkan akurasi model
        if score is not None:
            st.write("Model yang digunakan memiliki tingkat akurasi: {:.2f}%".format(score * 100))
