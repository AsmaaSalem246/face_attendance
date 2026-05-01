#  Face Recognition Attendance System

##  Project Overview

An AI-based system that automates attendance using face recognition.

##  Features

* Face Detection using MTCNN
* Face Recognition using KNN
* Image Preprocessing (lighting & noise)
* Automatic Attendance Logging
* Real-time camera support

##  Technologies Used

* Python
* OpenCV
* face_recognition
* MTCNN
* Scikit-learn
* Streamlit

##  How to Run

```bash
pip install -r requirements.txt
python encoding.py
streamlit run app.py
```

##  Output

* Recognizes faces in real-time
* Automatically records attendance (no duplicates per day)
