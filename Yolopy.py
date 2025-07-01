
import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import random
import numpy as np

# Page configuration
st.set_page_config(page_title="YOLOv8 Video Object Detection", layout="wide")

st.title("🎯 YOLOv8 Video Object Detection")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Load class names
with open(r"C:\Users\Nagesh\Downloads\coc.txt", "r") as myfile:
    class_list = myfile.read().split("\n")

# Generate random colors for each class
detection_colors = []
for _ in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load YOLOv8 model
model = YOLO(r"weights/yolov8n.pt")

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Video capture
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Cannot open video.")
    else:
        stframe = st.empty()
        run = st.checkbox("Start Detection")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("✅ Video processing complete.")
                break

            # Perform prediction
            results = model.predict(source=frame, conf=0.45, verbose=False)

            # Draw boxes and labels
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())

                    color = detection_colors[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_list[cls_id]}: {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            # Convert BGR to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            stframe.image(frame, channels="RGB")

        cap.release()
