# UI.py
import streamlit as st
from PIL import Image
import tempfile
import os
import math
import main
from compare import run as clip_run  


st.set_page_config(page_title="Near-Duplicate Image Detection", layout="wide")
st.title("🖼️ Near-Duplicate Image Detection")

# SIDEBAR 
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    top_k = st.slider("Top-K CLIP results", 1, 5, 5)
    run_search = st.button("🔍 Run Search")

#DISPLAY GRID 
def display_images_grid(images, scores=None, cols=3, image_width=300):
    import math
    import streamlit as st

    if not images:
        st.info("No images to display.")
        return

    rows = math.ceil(len(images) / cols)
    idx = 0
    for _ in range(rows):
        columns = st.columns(cols)
        for col in columns:
            if idx >= len(images):
                break
            with col:
                st.image(images[idx], width=image_width)
                if scores:
                    st.caption(f"{scores[idx]:.4f}")
            idx += 1



if run_search:
    if uploaded_file is None:
        st.warning("Please upload an image first!")
    else:
        query_image = Image.open(uploaded_file)
        st.subheader("Query Image")
        st.image(query_image, width=300)

        # Saving uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            query_image.save(temp_path)

        try:
            with st.spinner("Finding duplicates using pHash..."):
                direct_paths, direct_scores, ai_candidate_paths, ai_candidate_files = main.process_image(temp_path)

            # Displaying direct pHash duplicates
            st.subheader(f"✅ Direct pHash Matches ({len(direct_paths)})")
            display_images_grid(direct_paths, direct_scores, cols=3)

            
            if ai_candidate_files:
                st.subheader(f"CLIP/PCA Similarity (Top-{top_k})")
                with st.spinner("Computing CLIP embeddings..."):
                    clip_results_df = clip_run(temp_path, ai_candidate_files)
                    if clip_results_df is not None:
                        clip_paths = [
                            os.path.join(main.IMAGE_BASE, str(row.folder), str(row.file))
                            for _, row in clip_results_df.iterrows()
                        ]
                        clip_scores = clip_results_df["similarity"].tolist()
                        display_images_grid(clip_paths[:top_k], clip_scores[:top_k], cols=3)
                    else:
                        st.info("No similar images found by CLIP/PCA.")
            else:
                st.info("No images passed to CLIP/PCA after pHash filtering.")

        finally:
            os.remove(temp_path)
