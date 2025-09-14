import streamlit as st
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

# ----------------------------
# Load YOLO model (once)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # best.pt must be in same folder as app.py

model = load_model()

# Load OCR reader (once)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

ocr_reader = load_ocr()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Vehicle Number Plate Detection & OCR")
st.write("Upload an image to detect number plates and extract text.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Save to a temporary file YOLO can read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, image_bgr)
        temp_file_path = temp_file.name

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # Run YOLO Detection
    # ----------------------------
    results = model.predict(source=temp_file_path, conf=0.5, save=False)

    # Annotated image
    annotated_img = results[0].plot()
    annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    st.image(annotated_img_pil, caption="Detected Plates", use_column_width=True)

    # ----------------------------
    # OCR Extraction
    # ----------------------------
    st.subheader("Extracted Number Plate Texts")
    extracted_texts = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate_crop = image_bgr[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        ocr_result = ocr_reader.readtext(plate_crop_rgb)
        plate_text = " ".join([t[1] for t in ocr_result])
        extracted_texts.append(plate_text)

    if extracted_texts:
        for i, text in enumerate(extracted_texts, start=1):
            st.write(f"Plate {i}: {text}")
    else:
        st.write("No text detected on plates.")

    # Cleanup
    os.remove(temp_file_path)
