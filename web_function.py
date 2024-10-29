# Import modul yang digunakan
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import streamlit as st

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("data_tsunami_bersih.csv")

    x = df[["magnitude","cdi","mmi","alert","sig","net","nst","dmin","gap","magType","depth","latitude","longitude"]]
    y = df["tsunami"].values.ravel()

    return df, x, y

# Model dasar (base learners)
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('nn', MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]
# Model meta (meta-learner)
meta_learner = LogisticRegression(max_iter=1000)

# Fungsi untuk melatih model Stacking
@st.cache_data
def train_model(x, y):
    # Membuat model Stacking
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=-1)
    
    # Melatih model
    stacking_model.fit(x, y)
    
    # Menghitung skor pelatihan
    score = stacking_model.score(x, y)
    
    return stacking_model, score


# Fungsi untuk prediksi
def predict(x, y, features):
    model, score = train_model(x, y)
    
    # Melakukan prediksi pada fitur input
    prediction = model.predict(np.array(features).reshape(1, -1))
    
    return prediction, score


