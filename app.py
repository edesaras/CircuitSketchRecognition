import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO  # Make sure this import works in your Hugging Face environment
from io import BytesIO


@st.cache_resource
def load_model():
    """
        Load and cache the model
    """
    model = YOLO("weights.pt")  # Adjust path if needed
    return model

def predict(model, image, font_size, line_width):
    """
        Run inference and return annotated image  
    """
    results = model.predict(image)
    r = results[0]
    im_bgr = r.plot(conf=False, pil=True, font_size=font_size, line_width=line_width)  # Returns a PIL image if pil=True
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert BGR to RGB
    return im_rgb

def file_uploader_cb(uploaded_file, font_size, line_width):
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        # Display Uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    # Perform inference
    annotated_img = predict(model, image, font_size, line_width)
    with col2:
        # Display the prediction
        st.image(annotated_img, caption='Prediction', use_column_width=True)
    # write image to memory buffer for download
    imbuffer = BytesIO()
    annotated_img.save(imbuffer, format="JPEG")
    st.download_button("Download Annotated Image", data=imbuffer, file_name="Annotated_Sketch.jpeg", mime="image/jpeg", key="upload")

def image_capture_cb(capture, font_size, line_width, col):
    image = Image.open(capture).convert("RGB")
    # Perform inference
    annotated_img = predict(model, image, font_size, line_width)
    with col:
        # Display the prediction
        st.image(annotated_img, caption='Prediction', use_column_width=True)
    # write image to memory buffer for download
    imbuffer = BytesIO()
    annotated_img.save(imbuffer, format="JPEG")
    st.download_button("Download Annotated Image", data=imbuffer, file_name="Annotated_Sketch.jpeg", mime="image/jpeg", key="capture")

if __name__ == "__main__":
    # set page configurations and display/annotation options
    st.set_page_config(
        page_title="Circuit Sketch Recognizer",
        layout="wide"
    )
    st.title("Circuit Sketch Recognition")
    with st.sidebar:
        font_size = st.slider(label="Font Size", min_value=6, max_value=64, step=1, value=24)
        line_width = st.slider(label="Bounding Box Line Thickness", min_value=1, max_value=8, step=1, value=3)

    model = load_model()
    
    # user specifies to take/upload picture, view examples
    tabs = st.tabs(["Capture Picture", "Upload Your Image", "Show Examples"])
    with tabs[0]:
        # File uploader allows user to add their own image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_uploader_cb(uploaded_file, font_size, line_width)
    with tabs[1]:
        # Camera Input allows user to take a picture
        col1, col2 = st.columns(2)
        with col1:
            capture = st.camera_input("Take a picture with Camera")
        if capture is not None:
            image_capture_cb(capture, font_size, line_width, col2)
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.image('example1.jpg', use_column_width=True, caption='Example 1')
        with col2:
            st.image('example2.jpg', use_column_width=True, caption='Example 2')
