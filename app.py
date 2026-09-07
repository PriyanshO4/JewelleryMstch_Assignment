import os
import streamlit as st
from PIL import Image
from matcher import (
    load_model,
    load_dataset,
    get_image_embedding,
    precompute_earring_embeddings,
    find_matching_earrings,
)

# paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "candidate_dataset.csv")

st.set_page_config(page_title="Jewellery Matcher", layout="wide")
st.title("Jewellery Matcher")
st.markdown("Find matching earrings for a necklace using visual similarity.")


# ── load model once ──────────────────────────────────────────────
@st.cache_resource
def init_model():
    return load_model()


# ── load dataset once ────────────────────────────────────────────
@st.cache_data
def init_dataset():
    return load_dataset(CSV_PATH, IMAGE_DIR)


# ── precompute earring embeddings once ───────────────────────────
@st.cache_data
def init_earring_embeddings(_model, _processor, _device, earrings_df):
    return precompute_earring_embeddings(earrings_df, IMAGE_DIR, _model, _processor, _device)


model, processor, device = init_model()
necklaces, earrings = init_dataset()

if earrings.empty:
    st.error("No earring images found. Check the images/ folder and CSV.")
    st.stop()

earring_embeddings, earring_ids = init_earring_embeddings(model, processor, device, earrings)


# ── sidebar: choose input method ─────────────────────────────────
st.sidebar.header("Choose a Necklace")

input_method = st.sidebar.radio("Input method", ["Select from inventory", "Upload an image"])

necklace_img = None
necklace_label = None

if input_method == "Select from inventory":
    if necklaces.empty:
        st.sidebar.warning("No necklace images found in inventory.")
    else:
        necklace_options = {row["id"]: row["image_file"] for _, row in necklaces.iterrows()}
        selected_id = st.sidebar.selectbox("Necklace", list(necklace_options.keys()))
        necklace_img_path = os.path.join(IMAGE_DIR, necklace_options[selected_id])
        necklace_img = Image.open(necklace_img_path).convert("RGB")
        necklace_label = selected_id
        st.sidebar.image(necklace_img, caption=selected_id, use_container_width=True)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a necklace image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        necklace_img = Image.open(uploaded_file).convert("RGB")
        necklace_label = uploaded_file.name
        st.sidebar.image(necklace_img, caption="Uploaded image", use_container_width=True)

top_k = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=3)


# ── run matching ─────────────────────────────────────────────────
if st.sidebar.button("Find Matching Earrings", type="primary"):
    if necklace_img is None:
        st.warning("Please select or upload a necklace image first.")
    else:
        with st.spinner("Computing similarity..."):
            necklace_emb = get_image_embedding(necklace_img, model, processor, device)
            results = find_matching_earrings(necklace_emb, earring_embeddings, earring_ids, top_k)

        st.subheader(f"Top {len(results)} Matching Earrings for {necklace_label}")

        cols = st.columns(min(len(results), 5))
        for idx, (eid, score) in enumerate(results):
            col = cols[idx % len(cols)]
            ear_row = earrings[earrings["id"] == eid].iloc[0]
            ear_img_path = os.path.join(IMAGE_DIR, ear_row["image_file"])
            ear_img = Image.open(ear_img_path).convert("RGB")
            with col:
                st.image(ear_img, use_container_width=True)
                st.markdown(f"**{eid}**")
                st.caption(f"Similarity: {score:.4f}")
else:
    st.info("Select or upload a necklace image from the sidebar and click **Find Matching Earrings**.")
