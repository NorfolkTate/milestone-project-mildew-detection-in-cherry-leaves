import numpy as np
import streamlit as st

from src.data_management import load_image_file, resize_input_image, preprocess_input
from src.model_management import load_model_cached, CLASS_NAMES, IMG_SIZE 
# code inspired by stackoverflow and ref. in readme

def show():
    st.header("Predict Leaf Infection")
    file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if file is None:
        st.info("Please upload an image to continue.")
        return

    img = load_image_file(file)
    st.image(img, caption="Uploaded image", use_column_width=True)

    img_resized = resize_input_image(img, size=IMG_SIZE)
    x = preprocess_input(img_resized)

    model = load_model_cached()
    if model is None:
        st.error("Model not found. Place it at `outputs/models/cherry_leaf_model.keras`.")
        return

    probs = None  # code inspired by stackoverflow and ref. in readme

    try:
        probs = model.predict(x, verbose=0)

        if probs.ndim == 1:
            probs = np.expand_dims(probs, 0)
        if probs.shape[1] == 1:
            p1 = probs[:, 0]
            probs = np.vstack([1 - p1, p1]).T

    except Exception as e:
        st.error("Prediction failed. Check the model input size and preprocessing.")
        st.code(str(e))

    if probs is None:
        return

    pred_idx = int(np.argmax(probs[0]))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(np.max(probs[0]))
    # code explained by NumPy documentation and ref. in readme

    st.success(f"This leaf is predicted to be **{pred_label}** "
               f"with {confidence:.1%} confidence.")

    prob_map = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs[0])}
    st.subheader("Class probabilities")
    st.json(prob_map)

