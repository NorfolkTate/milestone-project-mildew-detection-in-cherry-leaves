import streamlit as st

def show():
    st.title("Cherry Leaf Mildew Detector")
    st.markdown("""
    ## Welcome  

    This interactive dashboard helps you explore and test a machine learning model 
    built to detect **powdery mildew** on cherry tree leaves.  
    Powdery mildew is a common fungal disease that can damage crops and reduce yields, 
    so being able to **quickly identify infected leaves** can support farmers and researchers.  

    ---

    ### How it works:
    - **Predict Leaf Infection**: Upload a cherry leaf image and the model will predict 
      whether the leaf is *healthy* or *infected*.  
    - **Visualise the Dataset**: Explore the images used to train the model, 
      including class balance and sample leaves.  
    - **Model Performance**: Review training metrics (accuracy & loss) and 
      understand how well the model generalises.  

    ---

    Use the sidebar on the left to navigate through the pages
    """)
