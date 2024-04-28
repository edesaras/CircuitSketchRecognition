import streamlit as st
from PIL import Image
from ultralytics import YOLO  # Make sure this import works in your Hugging Face environment
from io import BytesIO
import numpy as np
import pandas as pd 
from transformers import VisionEncoderDecoderModel, TrOCRProcessor 

@st.cache_resource
def load_ocr_model():
    """
        Load and cache the ocr model and processor
    """
    model = VisionEncoderDecoderModel.from_pretrained('edesaras/TROCR_finetuned_on_CSTA', cache_dir='./models/TrOCR')
    processor = TrOCRProcessor.from_pretrained("edesaras/TROCR_finetuned_on_CSTA", cache_dir='./models/TrOCR')
    return model, processor

@st.cache_resource
def load_model():
    """
        Load and cache the model
    """
    model = YOLO('./models/YOLO/weights.pt')
    return model

def predict(model, image, font_size, line_width):
    """
        Run inference and return annotated image  
    """
    results = model.predict(image)
    r = results[0]
    im_bgr = r.plot(conf=False, pil=True, font_size=font_size, line_width=line_width)  # Returns a PIL image if pil=True
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert BGR to RGB
    return im_rgb, r

def extract_text_patches(result, image):
    image = np.array(image)
    text_bboxes = []
    for i, label in enumerate([result.names[id.item()] for id in result.boxes.cls]):
        if label == 'text':
            bbox = result.boxes.xyxy[i]
            text_bboxes.append([round(i.item()) for i in bbox])
    crops = []
    for box in text_bboxes:
        xmin, ymin, xmax, ymax = box
        crop_img = image[ymin:ymax, xmin:xmax]
        crops.append(crop_img)
    return crops, text_bboxes

def ocr_predict(model, processor, crops):
    pixel_values = processor(crops, return_tensors="pt").pixel_values
    # Generate text with TrOCR
    generated_ids = model.generate(pixel_values)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts

def file_uploader_cb(model, ocr_model, ocr_processor, uploaded_file, font_size, line_width):
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        # Display Uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
    # Perform inference
    annotated_img, result = predict(model, image, font_size, line_width)
    with col2:
        # Display the prediction
        st.image(annotated_img, caption='Prediction', use_column_width=True)
    # write image to memory buffer for download
    imbuffer = BytesIO()
    annotated_img.save(imbuffer, format="JPEG")
    st.download_button("Download Annotated Image", data=imbuffer, file_name="Annotated_Sketch.jpeg", mime="image/jpeg", key="upload")
    
    st.subheader('Transcription')
    crops, text_bboxes = extract_text_patches(result, image)
    texts = ocr_predict(ocr_model, ocr_processor, crops)
    transcription_df = pd.DataFrame(zip(texts, *np.array(text_bboxes).T),  
             columns=['Transcription', 'xmin', 'ymin', 'xmax', 'ymax'])
    st.dataframe(transcription_df)

def image_capture_cb(model, ocr_model, ocr_processor, capture, font_size, line_width, col):
    image = Image.open(capture).convert("RGB")
    # Perform inference
    annotated_img, result = predict(model, image, font_size, line_width)
    with col:
        # Display the prediction
        st.image(annotated_img, caption='Prediction', use_column_width=True)
    # write image to memory buffer for download
    imbuffer = BytesIO()
    annotated_img.save(imbuffer, format="JPEG")
    st.download_button("Download Annotated Image", data=imbuffer, file_name="Annotated_Sketch.jpeg", mime="image/jpeg", key="capture")

    st.subheader('Transcription')
    crops, text_bboxes = extract_text_patches(result, image)
    texts = ocr_predict(ocr_model, ocr_processor, crops)
    transcription_df = pd.DataFrame(zip(texts, *np.array(text_bboxes).T),  
             columns=['Transcription', 'xmin', 'ymin', 'xmax', 'ymax'])
    st.dataframe(transcription_df)
