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

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load environment variables
load_dotenv(Path(".git/.env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_clip_embeddings(images_path, model):

    image_paths = glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
    
    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path)
        embedding = model.encode(image)
        embeddings.append(embedding)
    
    return embeddings, image_paths

def create_faiss_index(embeddings, image_paths, output_path):

    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    
    vectors = np.array(embeddings).astype(np.float32)

    # Add vectors to the index with IDs
    index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
    # Save the index
    faiss.write_index(index, output_path)
    print(f"Index created and saved to {output_path}")
    
    # Save image paths
    with open(output_path + '.paths', 'w') as f:
        for img_path in image_paths:
            f.write(img_path + '\n')
    
    return index

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f]
    print(f"Index loaded from {index_path}")
    return index, image_paths

def retrieve_similar_images(query, model, index, image_paths, top_k=3):
    
    # query preprocess:
    if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        query = Image.open(query)

    query_features = model.encode(query)
    query_features = query_features.astype(np.float32).reshape(1, -1)

    distances, indices = index.search(query_features, top_k)

    retrieved_images = [image_paths[int(idx)] for idx in indices[0]]

    return query, retrieved_images

IMAGES_PATH = 'D:\MME_RAG\image_database'
OUTPUT_INDEX_PATH = '/vector.index"

model = SentenceTransformer('clip-ViT-B-32')

embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model)

index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH)

index, image_paths = load_faiss_index(OUTPUT_INDEX_PATH)

query = 'quantum computer'

query, retrieved_images = retrieve_similar_images(query, model, index, image_paths, top_k=1)

visualize_results(query, retrieved_images)