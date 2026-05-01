import os
import cv2
import pickle
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from mtcnn import MTCNN
from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

detector = MTCNN()

def load_model():

    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError(
            "❌ encodings.pkl not found! Run encoding.py first"
        )

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["encodings"])
    y = np.array(data["names"])

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    return knn, y


def recognize_face(img):

    try:
        knn, labels = load_model()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(img_rgb)

        if len(results) == 0:
            return "No face found", img

        x, y_box, w, h = results[0]['box']

        x, y_box = abs(x), abs(y_box)

        face = img[y_box:y_box + h, x:x + w]

        face = preprocess_image(face)

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        enc = face_recognition.face_encodings(face_rgb)

        if len(enc) == 0:
            return "No encoding", img

        encoding = enc[0]

        distances, indices = knn.kneighbors([encoding])

        if distances[0][0] < 0.5:
            name = labels[indices[0][0]]
        else:
            name = "Unknown"

        cv2.rectangle(img, (x, y_box), (x + w, y_box + h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y_box - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return name, img

    except Exception as e:
        return f"Error: {str(e)}", img
