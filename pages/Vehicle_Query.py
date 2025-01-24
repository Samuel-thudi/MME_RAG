import streamlit as st
import os
import base64
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain

def image_encoding(file_path):
    """Convert image to base64 encoding"""
    with open(file_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    return image_base64

class Vehicle(BaseModel):
    Type: str = Field(
        ..., examples=["Car", "Truck", "Motorcycle", 'Bus', 'Van'],
        description="The type of the vehicle."
    )
    License: str = Field(
        ..., description="The license plate number of the vehicle. A continuous sequence of characters without dots, dashes, or spaces."
    )
    Make: str = Field(
        ..., examples=["Toyota", "Honda", "Ford", "Suzuki"],
        description="The Make of the vehicle."
    )
    Model: str = Field(
        ..., examples=["Corolla", "Civic", "F-150"],
        description="The Model of the vehicle."
    )
    Color: str = Field(
        ..., example=["Red", "Blue", "Black", "White"],
        description="Return the color of the vehicle."
    )

@chain
def create_prompt(inputs):
    """Create the prompt for processing."""
    prompt = [
        SystemMessage(content="""You are an AI assistant whose job is to inspect an image and provide the desired information from the image. If the desired field is not clear or not well detected, return None for this field. Do not try to guess."""),
        HumanMessage(
            content=[
                {"type": "text", "text": "Examine the main vehicle type, license plate number, make, model and color."},
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}", "detail": "low", }}
            ]
        )
    ]
    return prompt

@chain
def process_image(inputs):
    """Invoke the GPT model to extract information from the image."""
    model = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.0,
        max_tokens=1024,
    )
    output = model.invoke(inputs)
    return output.content

pipeline = create_prompt | process_image | JsonOutputParser(pydantic_object=Vehicle)

# Streamlit app
st.title("Vehicle Inspection App")

# File uploader
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_dir = r"\temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Encode the image
    image_base64 = image_encoding(file_path)  # Pass BytesIO directly
    
    # Display the uploaded image
    st.image(file_path, caption="Uploaded Vehicle Image", use_container_width =True)

    # Run the pipeline
    parser = JsonOutputParser(pydantic_object=Vehicle)
    instructions = parser.get_format_instructions()
    
    # Run the pipeline
    # try:
    output = pipeline.invoke({"image": image_base64})

    # Display the results
    st.subheader("Extracted Vehicle Details")
    st.write(output)

    # except Exception as e:
    #   st.error(f"An error occurred: {e}")

# # Optional: Batch processing section
# if st.checkbox("Batch Process Images from a Folder"):
#     folder_path = st.text_input("Enter the folder path:")

#     if folder_path and os.path.exists(folder_path):
#         image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

#         if image_paths:
#             batch_input = [{"image_path": path} for path in image_paths]
#             output = pipeline.batch(batch_input)

#             # Create a DataFrame to display the results
#             df = pd.DataFrame(output)
#             df.index = [os.path.basename(path) for path in image_paths]

#             st.dataframe(df)
#         else:
#             st.warning("No images found in the specified folder.")
#     else:
#         st.warning("Invalid folder path.")
