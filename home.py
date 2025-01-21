from openai import OpenAI
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
import tempfile

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load environment variables
load_dotenv(Path(".git/.env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CLIP model
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
client = OpenAI()

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

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

# image_query('Write a short label of what is show in this image?', image_path)

def main():
    # Set page configuration
    st.set_page_config(page_title="Image Info App", layout="wide")

    # Main app area
    st.title("Image Information Application")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as temp_file:
            temp_file.write(uploaded_image.read())
            temp_file_path = temp_file.name
        print(temp_file_path, '*'*200)
        # Load and display the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=300)  # Display image approximately 5x5 cm

    # Button 1: Get information about the image using ChatGPT
    if st.sidebar.button("Get Image Info (ChatGPT)"):
        answer = image_query('Write a short label of what is show in this image?', temp_file_path)
        st.write(answer)
        # st.info("Feature: Get information about the image using ChatGPT (functionality not implemented)")

    # Checkbox: Access file containing information on images in the database
    # access_file = st.sidebar.checkbox("Use Internal Database")
    # if access_file:
    #     st.success("Feature: Accessing database file (functionality not implemented)")

    # Button 2: Search for similar images using embeddings
    if st.sidebar.button("Search for Similar Images in Database"):
        st.info("Feature: Search for similar images using embeddings (functionality not implemented)")

    # Button 3: Get image info using database description
    if st.sidebar.button("Get Image Info from Database"):
        st.info("Feature: Get image information using the database description (functionality not implemented)")

if __name__ == "__main__":
    main()
