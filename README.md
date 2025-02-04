# Real-Time Object Detection

This project demonstrates real-time object detection using YOLOv5 with Streamlit for web-based image uploads and webcam feeds.

## Features

- **Image Upload**: Upload an image to detect objects.
- **Webcam Feed**: Real-time object detection using your webcam.

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Torch
- Pillow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/real-time-object-detection.git
    cd real-time-object-detection
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Image Upload

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload an image to see YOLOv5 in action.

### Webcam Feed

1. Run the Streamlit app:
    ```sh
    streamlit run webcam_app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Click the "Start" button to begin the webcam feed and "Stop" to end it.

## File Structure

- `app.py`: Streamlit app for image upload and object detection.
- `webcam_app.py`: Streamlit app for real-time object detection using webcam.
- `requirements.txt`: List of required Python packages.

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [Streamlit](https://streamlit.io/)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.