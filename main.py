import streamlit as st
from web_function import load_data
from Tabs import home, dataset, predict, visualise

Tabs = {
    "Home": home,
    "Dataset":dataset, 
    "Prediction": predict,
    "Visualisation": visualise
}

# membuat sidebar
st.sidebar.title("Navigasi")

# membuat radio option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Load dataset
df, x, y = load_data()

# kondisi call app function
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()
    
