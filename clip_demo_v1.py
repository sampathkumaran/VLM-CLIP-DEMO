import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import faiss
import pandas as pd
import os

# -------------------------------
# Load CLIP
# -------------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, processor, device


# -------------------------------
# Build embeddings + FAISS index
# -------------------------------
@st.cache_resource
def build_index(limit=1020):
    model, processor, device = load_clip()

    images, captions, img_embs, txt_embs = [], [], [], []
    df = pd.read_csv("data\Flowers102\captions.csv")
    captions = df['caption'].to_list()
    images_raw = df['image_path'].to_list()

    for index, img in enumerate(images_raw):
        curr = os.getcwd()
        im = os.path.join(curr, img)
        img = Image.open(im).convert('RGB')
        images.append(img)

        # Image embedding
        img_in = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_img = model.get_image_features(**img_in)
        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
        img_embs.append(emb_img.cpu().numpy())

        # Text embedding
        txt_in = processor(text=[captions[index]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb_txt = model.get_text_features(**txt_in)
        emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        txt_embs.append(emb_txt.cpu().numpy())

    img_embs = np.vstack(img_embs).astype("float32")
    txt_embs = np.vstack(txt_embs).astype("float32")

    # Build FAISS indices
    img_index = faiss.IndexFlatIP(img_embs.shape[1])
    img_index.add(img_embs)
    txt_index = faiss.IndexFlatIP(txt_embs.shape[1])
    txt_index.add(txt_embs)

    return images, captions, img_index, txt_index

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="CLIP + FAISS Retrieval", layout="wide")
st.title("🔍 CLIP + FAISS Retrieval Demo")

images, captions, img_index, txt_index = build_index(limit=1020)
model, processor, device = load_clip()

col1, col2 = st.columns(2)

# --- Text → Image ---
with col1:
    st.subheader("📝 Text → Image")
    query = st.text_input("Enter query", "A man riding a horse")
    k = st.slider("Top-K", 1, 5, 3)

    if st.button("Search Images"):
        txt_in = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            q_emb = model.get_text_features(**txt_in)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
        q_emb = q_emb.cpu().numpy()

        D, I = img_index.search(q_emb, k)
        st.write(f"**Query:** {query}")
        for idx in I[0]:
            st.image(images[idx], caption=captions[idx])

# --- Image → Text ---
with col2:
    st.subheader("🖼️ Image → Text")
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    k2 = st.slider("Top-K captions", 1, 5, 3)

    if upload is not None and st.button("Search Captions"):
        img = Image.open(upload).convert("RGB")
        st.image(img, caption="Query Image")
        img_in = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            q_emb = model.get_image_features(**img_in)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
        q_emb = q_emb.cpu().numpy()

        D, I = txt_index.search(q_emb, k2)
        st.write("**Top captions:**")
        for idx in I[0]:
            st.write("-", captions[idx])
