import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import torch
import torchvision.transforms as transforms
from utils import CNN

import joblib

# Load the pre-trained CNN model
our_model = joblib.load('./pretrained_model_joblib.pth')

# Define the transformation to apply to the user's drawing
'''transform = transforms.Compose([
transforms.Resize((3, 32, 32)),
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])'''
transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line")
#    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 20, 60, 40)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#fff")
bg_color = st.sidebar.color_picker("Background color hex: ")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Function to preprocess the user's drawing and make a prediction
def predict_digit(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = our_model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# Create the Streamlit app
def main():
    st.title("DB HW 2 - Digit Recognition App")
    st.write("Youngmok Kim, 20224017")
    st.write("This app uses a pre-trained CNN model to recognize hand-written digits.")
    st.write("Draw a digit between 0 and 9 and click the 'Confirm' button to make a prediction.")
    # Create a canvas for the user to draw on
    # Create a canvas component
    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Create a confirm button to make a prediction
    if st.button("Confirm"):
        # Convert the canvas image to PIL format
        image = canvas.image_data
        output = Image.fromarray(image.astype("uint8"), "RGB")
        # image = Image.fromarray(output)

        # Preprocess the image and make a prediction
        digit = predict_digit(output)

        # Display the predicted digit
        st.write("Predicted Digit:", digit)

# Run the app
if __name__ == "__main__":
    main()
