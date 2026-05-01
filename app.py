import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from model import recognize_face
from attendance import mark_attendance

st.set_page_config(page_title="Face Attendance System", layout="wide")

st.title("🎓 Face Recognition Attendance System")

# --------------------------
# 📌 INPUT MODE
# --------------------------
mode = st.radio("Choose mode:", ["Upload Image", "Live Camera"])

# --------------------------
# 📊 SHOW TODAY ATTENDANCE
# --------------------------
def show_attendance():
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    file_path = f"attendance_logs/{today}.csv"

    st.subheader("📊 Today's Attendance")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance yet today")

# --------------------------
# 📸 Upload Image
# --------------------------
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Input Image", channels="BGR")

        if st.button("Recognize"):

            name, processed = recognize_face(img)

            st.image(processed, channels="BGR")

            st.write(f"### 👤 {name}")

            if name not in ["Unknown", "No face found", "No encoding"]:
                msg = mark_attendance(name)
                st.success(msg)
            else:
                st.error("Face not recognized ❌")

# --------------------------
# 🎥 Live Camera
# --------------------------
elif mode == "Live Camera":

    run = st.checkbox("Start Camera")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        # 👇 الحل هنا (Mirror)
        frame = cv2.flip(frame, 1)

        name, processed = recognize_face(frame)

        if name not in ["Unknown", "No face found", "No encoding"]:
            mark_attendance(name)

        FRAME.image(processed, channels="BGR")

    cap.release()
st.divider()
show_attendance() 