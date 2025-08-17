from pathlib import Path
import streamlit as st

CLASS_NAMES = ("healthy", "powdery_mildew")
IMG_SIZE = (128, 128)
MODEL_PATH = Path("outputs/models/cherry_leaf_model.keras")

@st.cache_resource
def load_model_cached():
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None
    if not MODEL_PATH.exists():
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None
