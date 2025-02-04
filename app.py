import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image


# Use st.cache_resource for model caching
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


model = load_model()

st.title("Real-Time Object Detection Web App")
st.write("Upload an image to see YOLOv5 in action!")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image with PIL and convert to RGB
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Display the original uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to BGR for OpenCV drawing (since OpenCV works in BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    annotated_image = image_bgr.copy()

    with st.spinner("Detecting objects..."):
        results = model(image_np)

    # Get detections as a pandas DataFrame
    detections = results.pandas().xyxy[0]

    # Loop through detections and draw bounding boxes and labels manually
    for _, row in detections.iterrows():
        xmin, ymin = int(row['xmin']), int(row['ymin'])
        xmax, ymax = int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        # Draw rectangle and label on the image
        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            f"{label} {confidence:.2f}",
            (xmin, max(ymin - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Convert the annotated image back to RGB for display in Streamlit
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    st.image(annotated_image_rgb, caption='Detected Objects', use_column_width=True)

    # Optionally, display the detection details
    st.subheader("Detection Results")
    for _, row in detections.iterrows():
        st.write(
            f"**{row['name']}**: Confidence {row['confidence']:.2f} at "
            f"[({row['xmin']:.0f}, {row['ymin']:.0f}) - ({row['xmax']:.0f}, {row['ymax']:.0f})]"
        )
