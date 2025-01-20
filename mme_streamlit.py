import streamlit as st
import faiss
import json
import torch
import clip
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import base64
import openai
import openai

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load environment variables
load_dotenv(Path(".git/.env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CLIP model
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def image_query(query, image_path):
    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                },
                }
            ],
            }
        ],
        max_tokens=300,
    )
    # Extract relevant features from the response
    return response.choices[0].message.content

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

# Function to process images and extract features
def get_features_from_image(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_features

# Function to get similar images
def find_similar_images(query_image_features, index, k=2):
    distances, indices = index.search(query_image_features.cpu().numpy(), k)
    return list(zip(indices[0], distances[0]))

# Initialize FAISS index
def initialize_faiss_index(image_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor).float()

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features.cpu().numpy())
    return index, image_paths

# Streamlit app setup
st.title("Image Query and Search App")

# Sidebar for file upload and query
st.sidebar.header("Upload and Query")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg"])
user_query = st.sidebar.text_area("Enter your query", placeholder="What is the purpose of this item?")
query_button = st.sidebar.button("Submit Query")
similarity_button = st.sidebar.button("Find Similar Images")

# Directory containing database images
image_database_dir = "image_database/"
description_file = "description.json"

# Load FAISS index and image paths
st.spinner("Initializing search index...")
index, image_paths = initialize_faiss_index(image_database_dir)

# Display uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    image = Image.open(uploaded_image).convert("RGB")

    # Extract features from the uploaded image
    query_features = get_features_from_image(image)

    similar_image_path = None

    if similarity_button:
        # Perform similarity search
        similar_images = find_similar_images(query_features, index)

        st.subheader("Similar Images")
        for idx, distance in similar_images:
            similar_image_path = image_paths[idx]
            st.image(similar_image_path, caption=f"Similarity: {distance:.2f}", use_container_width=True)

    # Generate and display response if query button is pressed
    if query_button:
        # Load descriptions from JSON
        with open(description_file, 'r') as f:
            descriptions = [json.loads(line) for line in f]

        # Find the matching description
        description = None
        if similar_image_path:
            for entry in descriptions:
                if entry['image_path'] == similar_image_path:
                    description = entry['description']
                    break

        # Generate and display response from GPT
        if user_query and description:
            prompt = f"""
            Below is a user query. Answer the query using the description and image provided.

            User query:
            {user_query}

            Description:
            {description}
            """

            # Generate response (placeholder for now)
            response = image_query('Write a short label of what is show in this image?', image_path)
            st.subheader("AI Response")
            st.write(response)
