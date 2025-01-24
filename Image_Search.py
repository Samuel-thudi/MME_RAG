import streamlit as st
import glob
import faiss
import os
import json
import numpy as np
import openai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".git" / ".env")

# Initialize OpenAI API key (if needed)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
IMAGES_PATH = 'D:/MME_RAG/data/image_database'
OUTPUT_INDEX_PATH = 'vector.index'

# Load the SentenceTransformer CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Streamlit app title
st.title("Image Similarity Search with FAISS and CLIP")

# @st.cache_data
def generate_clip_embeddings(images_path, model):
    image_paths = glob.glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
    if not image_paths:
        raise FileNotFoundError(f"No images found in the directory: {images_path}")
    
    embeddings = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')  # Convert to RGB for compatibility
            embedding = model.encode(image)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    return embeddings, image_paths

# def generate_clip_embeddings(images_path, model):
#     """Generate embeddings for images in a given directory."""
#     image_paths = glob.glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
#     embeddings = []
#     for img_path in image_paths:
#         image = Image.open(img_path).convert('RGB')
#         embedding = model.encode(image)
#         embeddings.append(embedding)
#     return embeddings, image_paths

# @st.cache_data
def create_faiss_index(embeddings, image_paths, output_path):
    """Create and save a FAISS index from the embeddings."""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    vectors = np.array(embeddings).astype(np.float32)
    index.add_with_ids(vectors, np.array(range(len(embeddings))))
    faiss.write_index(index, output_path)
    with open(output_path + '.paths', 'w') as f:
        for img_path in image_paths:
            f.write(img_path + '\n')
    return index

# @st.cache_data
def load_faiss_index(index_path):
    """Load a FAISS index and associated image paths."""
    index = faiss.read_index(index_path)
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f]
    return index, image_paths

# @st.cache_data
def retrieve_similar_images(query, model, index, image_paths, top_k=3):
    """Retrieve similar images from the index based on a query."""
    if isinstance(query, Image.Image):
        query_features = model.encode(query)
    else:
        query_features = model.encode([query])[0]

    query_features = query_features.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_features, top_k)
    retrieved_images = [image_paths[int(idx)] for idx in indices[0]]
    return retrieved_images

# Generate or load embeddings and FAISS index
if not os.path.exists(OUTPUT_INDEX_PATH):
    st.info("Generating embeddings and creating FAISS index. This may take a while.")
    embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model)
    index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH)
else:
    st.success("Loading existing FAISS index.")
    index, image_paths = load_faiss_index(OUTPUT_INDEX_PATH)

# Query input (image or text)
query_type = st.radio("Query Type", ["Image", "Text"])

if query_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert('RGB')
        st.image(query_image, caption="Uploaded Query Image", use_container_width = True)
        retrieved_images = retrieve_similar_images(query_image, model, index, image_paths, top_k=3)
        st.header("Retrieved Images")
        for img_path in retrieved_images:
            st.image(Image.open(img_path), caption=img_path, use_container_width=True)

elif query_type == "Text":
    text_query = st.text_input("Enter a text query", placeholder="e.g., a cat sitting on a couch")
    if text_query:
        retrieved_images = retrieve_similar_images(text_query, model, index, image_paths, top_k=3)
        st.header("Retrieved Images")
        for img_path in retrieved_images:
            st.image(Image.open(img_path), caption=img_path, use_container_width=True)
