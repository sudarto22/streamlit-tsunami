import streamlit as st
import pandas as pd

def app():
    #judul halaman aplikasi
    #st.title("Data Set Gempa Bumi")
    st.markdown("<h1 style='text-align: center;'>Data Set Gempa Bumi</h1>", unsafe_allow_html=True)
    data = pd.read_csv("earthquake_data.csv")
    st.markdown("<h4 style='text-align: center;'>Data Gempa Bumi Sebelum di Preprosesing</h4>", unsafe_allow_html=True)
    st.dataframe(data)
    st.write("Jumlah Record:", data.shape[0])
    st.write("Jumlah Atribut:", data.shape[1])

   
    if st.button("Dataset Cleaned"):
       clean_data = pd.read_csv("data_gempa_clean.csv")
       st.markdown("<h4 style='text-align: center;'>Data Gempa Bumi Cleaned</h4>", unsafe_allow_html=True)
       st.dataframe(clean_data)
       st.write("Jumlah Record Setelah Cleaning:", clean_data.shape[0])
       st.write("Jumlah Atribut Setelah Cleaning:", clean_data.shape[1])
