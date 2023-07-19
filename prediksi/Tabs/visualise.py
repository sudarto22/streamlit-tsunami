import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import streamlit as st
from web_function import train_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


warnings.filterwarnings("ignore", category=DeprecationWarning)

def app(df, x, y):
    st.title("Visualisasi prediksi tsunami")

    if st.checkbox("Plot Confusion Matrix"):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model, score = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
       
        plt.figure(figsize=(10, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no_tsunami', 'tsunami'])
        disp.plot(include_values=True, cmap='Blues', ax=plt.gca())
        st.pyplot(plt.gcf())
    
    if st.checkbox("Show Performance Metrics"):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model, score = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=['no_tsunami', 'tsunami'])
        
        st.write("Performance Metrics:")
        st.code(report, language='text')

    if st.checkbox("Plot Decision Tree"):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model, score = train_model(x_train, y_train)
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_tree(model, feature_names=x.columns.tolist(), class_names=['no_tsunami', 'tsunami'], filled=True, ax=ax, fontsize=10)
        plt.tight_layout()  # Adjust the layout of the plot
        st.pyplot(fig)
