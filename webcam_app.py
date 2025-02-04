import cv2
import streamlit as st
import numpy as np
import torch
import time

# -----------------------------
# Helper functions and session state management

# Initialize the run flag in session state if not already present.
if "run" not in st.session_state:
    st.session_state["run"] = False

def start_stream():
    st.session_state["run"] = True

def stop_stream():
    st.session_state["run"] = False

# -----------------------------
# Load your YOLOv5 model (cached)
@st.cache_resource
def load_model():
    # Using the small YOLOv5 model for speed
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# -----------------------------
# Title and control buttons
st.title("Webcam Object Detection (OpenCV Loop)")

col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        start_stream()
with col2:
    if st.button("Stop"):
        stop_stream()

# -----------------------------
# Create an empty placeholder for the video frames
frame_placeholder = st.empty()

# -----------------------------
# Open webcam (0 is the default camera index)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")

# -----------------------------
# Main loop: process frames while run flag is True
while st.session_state["run"]:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (model & display format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the YOLOv5 model on the frame
    results = model(frame_rgb)
    # Get detections as a pandas DataFrame
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes and labels on the frame (on the RGB image)
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        # Draw a rectangle and label
        cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame_rgb,
            f"{label} {confidence:.2f}",
            (xmin, max(ymin - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Update the placeholder with the new frame
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Sleep briefly to control the update rate (adjust as needed)
    time.sleep(0.03)

# -----------------------------
# Cleanup: release the webcam when done
cap.release()
st.write("Stream stopped.")
