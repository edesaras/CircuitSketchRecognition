---
title: CircuitSketchRecognition
emoji: 👁
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: 1.33.0
app_file: 🤗_Hello.py
pinned: false
license: mit
---

# Circuit Sketch Recognition 👁

## About the App
The **Circuit Sketch Recognition** app demonstrates the power of AI to recognize hand-drawn circuit diagrams. By leveraging advanced models like TrOCR for text recognition and YOLOv8 for component detection, this app showcases how easily computers can understand sketches (Fine tuned on a CGHD-2304 dataset).

<p align="center">
  <img src="media/capture.gif" alt="Capture GIF" width="45%"/>
  <img src="media/upload.gif" alt="Upload GIF" width="45%"/>
</p>


## Features
- **Upload or Capture**: Upload a picture of your circuit sketch or use your camera to capture one in real-time.
- **Dual Display**: View both capture and upload options in a side-by-side format for easy access.
- **Example Gallery**: Explore a gallery of recognized sketches to see the accuracy and capabilities of our AI models.

## Deployment

Deployed on Streamlit and Hugging Face Spaces:

<a href="https://circuitsketchrecognition.streamlit.app/">
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit" width="30"/> Streamlit App
</a>

<a href="https://huggingface.co/spaces/edesaras/CircuitSketchRecognition">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" alt="Huggingface" width="30"/> Huggingface App
</a>

## Getting Started
To get started with the Circuit Sketch Recognition app, simply clone the repository and run the `🤗_Hello.py` file with Streamlit:

```bash
git clone https://github.com/edesaras/CircuitSketchRecognition.git
streamlit run 🤗_Hello.py