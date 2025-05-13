import streamlit as st

def show():
    st.title("Cherry Leaf Mildew Detector")
    st.markdown("""
    ### Project Overview
    This dashboard allows users to predict whether a cherry leaf is healthy or infected with powdery mildew.

    **Features:**
    - Predict leaf infection from images.
    - Visualise the dataset.
    - View model performance metrics.
    """)
