from pathlib import Path
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_management import iter_image_paths, DATA_ROOT
from src.model_management import CLASS_NAMES

DATA_ROOT = Path("inputs/dataset")

def show():
    st.header("Visualise Dataset")
    st.markdown (""" Choose which dataset you would like to analyse
    """)

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
        # code inspired by geeks for geeks and streamlit documentation and ref.in readme

        st.markdown(
        """
        **Graph: distribution of images per class in the selected dataset split.**  
        This chart shows whether the dataset is balanced between *healthy* and *powdery mildew* leaves.  
        Balanced data helps the model learn fairly, while imbalance could cause biased predictions.
        """
)

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
        st.exception(e) # code inspired by rollbar and ref. in readme

    st.markdown(
    """
    **Figure: Sample images from the selected class.**  
    These examples illustrate the visual appearance of cherry leaves.  
    Healthy leaves typically have uniform green color, while powdery mildew leaves show white fungal patches.
    """
    )

    st.subheader("Analysis and Conclusions")

    st.markdown("""
        - **Image counts**: By comparing the number of images across classes and dataset
          splits (train/val/test), we can confirm whether the dataset is balanced.
          A balanced dataset reduces the risk of bias in the model.
        
        - **Sample images**: Visualising leaf samples helps confirm that the images are
          correctly labelled and that mildew is clearly and visibly different from healthy leaves
                
        - **Implication for the model**: If one class has noticeably fewer images, the
          model may overfit to the majority class. This means predictions could become
          biased, e.g. the model might predict "healthy" too often simply because it has
          seen more healthy examples.  

          In future improvements, **data augmentation** (generating extra training
          images by flipping, rotating, or adjusting brightness/contrast of existing
          samples) could artificially increase the variety and count of the minority
          class. Alternatively, **resampling techniques** could be used to help
          the model train on a more balanced dataset.

        """)
