# model imports
import faiss
import json
import torch
from openai import OpenAI
import torch.nn as nn
from torch.utils.data import DataLoader
import clip

# helper imports
from tqdm import tqdm
import json
import os
import numpy as np
import pickle
from typing import List, Union, Tuple

# visualisation imports
import matplotlib.pyplot as plt
from PIL import Image
import base64

# environment variables
from dotenv import load_dotenv
from pathlib import Path
import openai
import os


# Specify the path to your .env file
dotenv_path = Path(".gitignore/config/.env")  # Adjust the path if needed
load_dotenv(dotenv_path=dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# #load model on device. The device you are running inference/training on is either a CPU or GPU if you have.
# device = "cpu"
# model, preprocess = clip.load("ViT-B/32",device=device)

# def get_image_paths(directory: str, number: int = None) -> List[str]:
#     image_paths = []
#     count = 0
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpeg'):
#             image_paths.append(os.path.join(directory, filename))
#             if number is not None and count == number:
#                 return [image_paths[-1]]
#             count += 1
#     return image_paths

# direc = 'image_database/'
# image_paths = get_image_paths(direc)

# def get_features_from_image_path(image_paths):
#   images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths]
#   image_input = torch.tensor(np.stack(images))
#   with torch.no_grad():
#     image_features = model.encode_image(image_input).float()
#   return image_features

# image_features = get_features_from_image_path(image_paths)

# index = faiss.IndexFlatIP(image_features.shape[1])
# index.add(image_features)

# data = []
# image_path = 'train1.jpeg'

# with open('description.json', 'r') as file:
#     for line in file:
#         data.append(json.loads(line))
# def find_entry(data, key, value):
#     for entry in data:
#         if entry.get(key) == value:
#             return entry
#     return None

# im = Image.open(image_path)
# plt.imshow(im)
# plt.show()


# def encode_image(image_path):
#     with open(image_path, 'rb') as image_file:
#         encoded_image = base64.b64encode(image_file.read())
#         return encoded_image.decode('utf-8')

# def image_query(query, image_path):
#     response = client.chat.completions.create(
#         model='gpt-4-vision-preview',
#         messages=[
#             {
#             "role": "user",
#             "content": [
#                 {
#                 "type": "text",
#                 "text": query,
#                 },
#                 {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
#                 },
#                 }
#             ],
#             }
#         ],
#         max_tokens=300,
#     )
#     # Extract relevant features from the response
#     return response.choices[0].message.content

# image_query('Write a short label of what is show in this image?', image_path)


# image_search_embedding = get_features_from_image_path([image_path])
# distances, indices = index.search(image_search_embedding.reshape(1, -1), 2) #2 signifies the number of topmost similar images to bring back
# distances = distances[0]
# indices = indices[0]
# indices_distances = list(zip(indices, distances))
# indices_distances.sort(key=lambda x: x[1], reverse=True)


# #display similar images
# for idx, distance in indices_distances:
#     print(idx)
#     path = get_image_paths(direc, idx)[0]
#     im = Image.open(path)
#     plt.imshow(im)
#     plt.show()



# similar_path = get_image_paths(direc, indices_distances[0][0])[0]
# element = find_entry(data, 'image_path', similar_path)

# user_query = 'What is the capacity of this item?'
# prompt = f"""
# Below is a user query, I want you to answer the query using the description and image provided.

# user query:
# {user_query}

# description:
# {element['description']}
# """
# image_query(prompt, similar_path)