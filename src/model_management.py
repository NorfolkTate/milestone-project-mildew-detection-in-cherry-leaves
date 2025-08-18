from pathlib import Path
import streamlit as st

CLASS_NAMES = ("healthy", "powdery_mildew")
IMG_SIZE = (256, 256)
MODEL_PATH = Path("outputs/models/cherry_leaf_model.keras")

@st.cache_resource  # code helpfully provided by stackoverflow and ref. in readme 
def load_model_cached():
    try:
        from tensorflow.keras.models import load_model # code helpfully provided by geeks for geeks and ref in readme
    except Exception:
        return None
    if not MODEL_PATH.exists():
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None
