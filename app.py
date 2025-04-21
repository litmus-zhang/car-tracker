import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import datetime
from ultralytics import YOLO  # make sure you have installed ultralytics
from utils import store_today_data, load_sheet_data

# Initialize session state for streaming control, counts, and data storage flag
if "streaming" not in st.session_state:
    st.session_state.streaming = True
if "car_count" not in st.session_state:
    st.session_state.car_count = 0
if "bus_count" not in st.session_state:
    st.session_state.bus_count = 0
if "truck_count" not in st.session_state:
    st.session_state.truck_count = 0
if "data_stored" not in st.session_state:
    st.session_state.data_stored = False  # ensures daily data is stored only once


def stop_streaming():
    st.session_state.streaming = False


def resume_streaming():
    st.session_state.streaming = True


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

    # Use a placeholder container for live metrics update
    metric_container = st.empty()
    mcols = metric_container.columns(3)
    mcols[0].metric("Car", st.session_state.car_count)
    mcols[1].metric("Bus", st.session_state.bus_count)
    mcols[2].metric("Truck", st.session_state.truck_count)

with col2:
    st.subheader("Camera Stream")
    st.write("Camera stream will be displayed here.")

    # Start streaming from the USB camera using OpenCV
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()

    # Display appropriate button based on the streaming state
    if st.session_state.streaming:
        st.button("Stop Streaming", on_click=stop_streaming)
    else:
        st.button("Resume Streaming", on_click=resume_streaming)

    # Run the streaming loop if enabled
    while cap.isOpened() and st.session_state.streaming:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)
        if results and results[0].boxes is not None:
            # Get detected class indices and update counts accordingly
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            for cls in classes:
                label = model.model.names[cls]
                if label == "car":
                    st.session_state.car_count += 1
                elif label == "bus":
                    st.session_state.bus_count += 1
                elif label == "truck":
                    st.session_state.truck_count += 1

        # Update live metrics in the left column
        mcols = metric_container.columns(3)
        mcols[0].metric("Car", st.session_state.car_count)
        mcols[1].metric("Bus", st.session_state.bus_count)
        mcols[2].metric("Truck", st.session_state.truck_count)

        # Optionally, draw detected boxes on the frame.
        annotated_frame = results[0].plot() if results else frame
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(annotated_frame, channels="RGB")
        time.sleep(0.03)  # Small delay to control frame rate

        # Check if it's the end of the day (23:59 or later) and data hasn't been stored yet.
        now = datetime.datetime.now().time()
        if now.hour == 23 and now.minute >= 59 and not st.session_state.data_stored:
            store_today_data(
                st.session_state.car_count,
                st.session_state.bus_count,
                st.session_state.truck_count,
            )
            st.session_state.data_stored = True
            st.success("Today's data has been automatically stored to Google Sheets.")

    cap.release()
