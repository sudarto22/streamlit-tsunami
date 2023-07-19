# Import modul yang digunakan
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("data_gempa_clean.csv")

    x = df[["magnitude","depth","date_time","alert","latitude","longitude"]]
    y = df["tsunami"].values.ravel()

    return df, x, y



from sklearn.model_selection import train_test_split

def train_model(x, y):
    # Membagi dataset menjadi data pelatihan dan data uji
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Membuat objek DecisionTreeClassifier dengan parameter-parameter yang ditentukan
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=3, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, random_state=42, splitter='best')
    
    # Melatih model menggunakan data pelatihan
    model.fit(x_train, y_train)
    
    # Menghitung skor akurasi model pada data uji
    score = model.score(x_test, y_test)
    
    # Mengembalikan model dan skor akurasi
    return model, score


#@st.cache_data
def predict(x, y, features):
    model, score = train_model(x, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score



