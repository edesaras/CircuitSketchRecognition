import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO  # Make sure this import works in your Hugging Face environment

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = YOLO("weights.pt")  # Adjust path if needed
    return model

model = load_model()

st.title("Circuit Sketch Recognition")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Perform inference
    results = model.predict(uploaded_file)
    r = results[0]
    im_bgr = r.plot(conf=False, pil=True, font_size=32, line_width=2)  # Returns a PIL image if pil=True
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert BGR to RGB

    # Display the prediction
    st.image(im_rgb, caption='Prediction', use_column_width=True)

# Optionally, display pre-computed example images
if st.checkbox('Show Example Results'):
    st.image(['example1.jpg', 'example2.jpg'], width=300, caption=['Example 1', 'Example 2'])
