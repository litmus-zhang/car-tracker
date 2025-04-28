import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import datetime
from ultralytics import YOLO  # make sure you have installed ultralytics
from utils import store_today_data, load_sheet_data
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize session state for streaming control, counts, and data storage flag
if "car_count" not in st.session_state:
    st.session_state.car_count = 0
if "bus_count" not in st.session_state:
    st.session_state.bus_count = 0
if "truck_count" not in st.session_state:
    st.session_state.truck_count = 0
if "data_stored" not in st.session_state:
    st.session_state.data_stored = False  # ensures daily data is stored only once

# Load YOLOv8 model (using a lightweight pretrained model for speed)
model = YOLO("yolov8n.pt")

st.header("Vehicle Detection App")
st.subheader("Detect vehicles in a video stream ")
st.markdown(
    """
    This app uses YOLOv8 and DeepSORT for vehicle detection and tracking.
    - **YOLOv8**: A state-of-the-art object detection model.
    - **DeepSORT**: A tracking algorithm that associates detected objects across frames.
    """
)

# Prepare the two columns for the UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detected Vehicles")
    # Load historical data from Google Sheets and plot the bar chart.
    sheet_df = load_sheet_data()
    if sheet_df.empty:
        st.write("No historical data available yet.")
    else:
        st.bar_chart(sheet_df.set_index("Date"), use_container_width=True)
        st.write("Detected vehicles over time.")

    # Display live metrics
    metric_container = st.empty()
    mcols = metric_container.columns(3)
    mcols[0].metric("Car", st.session_state.car_count)
    mcols[1].metric("Bus", st.session_state.bus_count)
    mcols[2].metric("Truck", st.session_state.truck_count)


# Define a VideoTransformer for processing the camera stream
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert video frame to numpy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        # Run YOLOv8 inference on the frame
        results = model(img)
        if results and results[0].boxes is not None:
            # Update counts and annotate frame
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            for cls in classes:
                label = model.model.names[cls]
                if label == "car":
                    st.session_state.car_count += 1
                elif label == "bus":
                    st.session_state.bus_count += 1
                elif label == "truck":
                    st.session_state.truck_count += 1
            annotated = results[0].plot()
        else:
            annotated = img
        # Convert annotated frame to RGB before returning
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated


with col2:
    st.subheader("Camera Stream")
    st.write("Camera stream will be displayed here using streamlit_webrtc.")
    webrtc_streamer(key="example", video_transformer_factory=YOLOVideoTransformer)

# Optionally, check and store end-of-day data
now = datetime.datetime.now().time()
if now.hour == 23 and now.minute >= 59 and not st.session_state.data_stored:
    store_today_data(
        st.session_state.car_count,
        st.session_state.bus_count,
        st.session_state.truck_count,
    )
    st.session_state.data_stored = True
    st.success("Today's data has been automatically stored to Google Sheets.")
