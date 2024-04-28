import streamlit as st
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import load_model, file_uploader_cb, load_ocr_model

if __name__ == "__main__":
    # set page configurations and display/annotation options
    st.set_page_config(
        page_title="Circuit Sketch Recognizer",
        layout="wide"
    )
    
    with st.sidebar:
        font_size = st.slider(label="Font Size", min_value=6, max_value=64, step=1, value=24)
        line_width = st.slider(label="Bounding Box Line Thickness", min_value=1, max_value=8, step=1, value=3)

    model = load_model()
    ocr_model, ocr_processor = load_ocr_model()

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_uploader_cb(model, ocr_model, ocr_processor, uploaded_file, font_size, line_width)
