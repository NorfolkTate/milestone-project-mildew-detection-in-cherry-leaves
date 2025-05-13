import streamlit as st
from app_pages import home, model_prediction, visualiser, model_performance

st.set_page_config(page_title="Cherry Leaf Mildew Detector", layout="wide")

# Sidebar Navigation
PAGES = {
    "Home": home,
    "Predict Leaf Infection": model_prediction,
    "Visualise Dataset": visualiser,
    "Model Performance": model_performance
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.show()
