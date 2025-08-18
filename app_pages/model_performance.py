import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

from src.model_management import load_model_cached, CLASS_NAMES


def show():
    st.header("Model Performance")

    history_path = Path("outputs/models/training_history.pkl")
    if not history_path.exists():
        st.error("No training history found. Please ensure 'outputs/models/training_history.pkl' exists.")
        return

    with open(history_path, "rb") as f: # code inspired by python documentation and geeks and geeks and ref. in readme
        history = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # accuracy
    axes[0].plot(history["accuracy"], label="Train acc") # code inspired by stackoverflow and ref. in readme
    axes[0].plot(history["val_accuracy"], label="Val acc")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # loss
    axes[1].plot(history["loss"], label="Train loss")
    axes[1].plot(history["val_loss"], label="Val loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    st.pyplot(fig)

    st.subheader("Analysis & Conclusions")

    st.markdown("""
        - **Accuracy:** Validation accuracy reached **100% across all epochs**.  
          This indicates the model can perfectly distinguish between classes in the validation set.  
        - **Loss:** Training and validation loss both decreased towards zero, showing the model learned effectively.  
        - **Conclusion:** The model appears highly effective, which is very encouraging and therefore tempting to lean towards a perfect success. However, critically thinking, the unusually high accuracy suggests there is signs of the model overfitting it should be 
          tested further on the **test set** and real-world data to confirm robustness and avoid overfitting concerns.
        """)
