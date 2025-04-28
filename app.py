import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
import threading
from ultralytics import YOLO  # ensure ultralytics is installed
from utils import store_today_data, load_sheet_data
from streamlit_webrtc import webrtc_streamer
import av

# --- Global counter and lock (do not update st.session_state directly in video threads) ---
if "global_counts" not in st.session_state:
    st.session_state.global_counts = {"car": 0, "bus": 0, "truck": 0}
counter_lock = threading.Lock()

# --- Load YOLOv8 model ---
model = YOLO("yolov8n.pt")

st.header("Vehicle Detection App")
st.subheader("Detect vehicles in a video stream")
st.markdown(
    """
    This app uses YOLOv8 and DeepSORT for vehicle detection and tracking.
    - **YOLOv8**: A state-of-the-art object detection model.
    - **DeepSORT**: A tracking algorithm that associates detected objects across frames.
    """
)

# --- UI Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detected Vehicles")
    # Load historical data from Google Sheets and plot bar chart
    sheet_df = load_sheet_data()
    if sheet_df.empty:
        st.write("No historical data available yet.")
    else:
        st.bar_chart(sheet_df.set_index("Date"), use_container_width=True)
        st.write("Detected vehicles over time.")

    # Placeholder for live metrics (to be updated periodically)
    metrics_placeholder = st.empty()


# --- Video frame transform callback using streamlit_webrtc ---
def transform(frame):
    # Convert incoming frame to ndarray (BGR)
    img = frame.to_ndarray(format="bgr24")
    results = model(img)
    if results and results[0].boxes is not None:
        # Update global counts (thread-safe)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            label = model.model.names[cls]
            if label in st.session_state.global_counts:
                with counter_lock:
                    st.session_state.global_counts[label] += 1
        # Annotate frame (draw boxes, etc.)
        annotated = results[0].plot()
    else:
        annotated = img
    # Return new frame in the correct format
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# --- Start streamlit_webrtc ---
with col2:
    st.subheader("Camera Stream")
    st.write("Camera stream will be displayed here using streamlit_webrtc.")
    webrtc_streamer(
        key="example",
        video_frame_callback=transform,
        rtc_configuration={
            "max_frames_per_second": 30,
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                    ]
                }
            ],
        },
    )


# --- Background thread to update metrics display periodically ---
def update_metrics():
    while True:
        with counter_lock:
            car = st.session_state.global_counts["car"]
            bus = st.session_state.global_counts["bus"]
            truck = st.session_state.global_counts["truck"]
        mcols = metrics_placeholder.columns(3)
        mcols[0].metric("Car", car)
        mcols[1].metric("Bus", bus)
        mcols[2].metric("Truck", truck)
        time.sleep(1)  # update every second


threading.Thread(target=update_metrics, daemon=True).start()

# --- Optionally, store end-of-day data ---
now = datetime.datetime.now().time()
if now.hour == 23 and now.minute >= 59:
    store_today_data(
        car_count=st.session_state.global_counts["car"],
        bus_count=st.session_state.global_counts["bus"],
        truck_count=st.session_state.global_counts["truck"],
    )
    st.success("Today's data has been automatically stored to Google Sheets.")
