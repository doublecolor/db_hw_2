import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load the pre-trained CNN model
model = torch.load('pretrained_model.pth')
model.eval()

# Define the transformation to apply to the user's drawing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to preprocess the user's drawing and make a prediction
def predict_digit(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# Create the Streamlit app
def main():
    st.title("Digit Recognition App")

    # Create a canvas for the user to draw on
    canvas = st.sketchpad()

    # Create a reset button to clear the canvas
    if st.button("Reset"):
        canvas.clear()

    # Create a confirm button to make a prediction
    if st.button("Confirm"):
        # Convert the canvas image to PIL format
        image = Image.fromarray(canvas.to_image())

        # Preprocess the image and make a prediction
        digit = predict_digit(image)

        # Display the predicted digit
        st.write("Predicted Digit:", digit)

# Run the app
if __name__ == "__main__":
    main()