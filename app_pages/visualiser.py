from pathlib import Path
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.model_management import CLASS_NAMES
from src.data_management import iter_image_paths, load_image_path

DATA_ROOT = Path("inputs/dataset")

def show():
    st.header("Visualise Dataset")

    try:
        split = st.selectbox("Dataset split", ["train", "val", "test"])
        split_dir = DATA_ROOT / split

        if not split_dir.exists():
            st.error(f"Split foldr not found: {split_dir}")
            return

        counts = {}
        for cls in CLASS_NAMES:
            folder = split_dir / cls
            counts[cls] = sum(1 for _ in iter_image_paths(folder)) if folder.exists() else 0

        fig, ax = plt.subplots()
        ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_ylabel("# images")
        st.pyplot(fig)

        st.subheader("Sample images")
        cls = st.selectbox("Class", CLASS_NAMES, key="vis_cls")
        n = st.slider("How many samples?", 4, 20, 8, step=4)

        cls_dir = split_dir / cls
        paths = list(iter_image_paths(cls_dir)) if cls_dir.exists() else []

        if not paths:
            st.info(f"No images found in {cls_dir}")
            return

        random.shuffle(paths)
        paths = paths[:n]

        cols = st.columns(4)
        for i, p in enumerate(paths):
            with cols[i % 4]:
                st.image(str(p), use_container_width=True)

    except Exception as e:
        st.error("Error while rendering visualiser page.")
        st.exception(e)