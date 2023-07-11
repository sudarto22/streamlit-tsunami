import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
import streamlit as st
from sklearn.tree import plot_tree
from web_function import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=DeprecationWarning)

def app(df, x, y):
    st.title("Visualisasi prediksi tsunami")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        class_names = ['no_tsunami', 'tsunami']  # Ubah sesuai dengan label yang digunakan dalam dataset Anda
       
        plt.figure(figsize=(10, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no_tsunami', 'tsunami'])
        disp.plot(include_values=True, cmap='Blues', ax=plt.gca())
        st.pyplot(plt.gcf())
    if st.checkbox("Show Performance Metrics"):
        model, score = train_model(x, y)
        y_pred = model.predict(x)
        report = classification_report(y, y_pred, target_names=['no_tsunami', 'tsunami'])
        
        st.write("Performance Metrics:")
        st.code(report, language='text')

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        fig, ax = plt.subplots(figsize=(10, 8))
        tree.plot_tree(model, feature_names=x.columns.tolist(), class_names=['no_tsunami', 'tsunami'], filled=True, ax=ax, fontsize=10)
        plt.tight_layout()  # Adjust the layout of the plot
        st.pyplot(fig)
        

 
