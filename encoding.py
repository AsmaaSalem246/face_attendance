import os
import cv2
import face_recognition
import pickle
from mtcnn import MTCNN
from preprocessing import preprocess_image

dataset_path = r"C:\Users\gomaa\face_attendance\Train_lfw"

detector = MTCNN()

known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_path):

        for image_name in os.listdir(person_path):

            if image_name.endswith((".jpg", ".png", ".jpeg")):

                image_path = os.path.join(person_path, image_name)
                img = cv2.imread(image_path)

                # 🔥 MTCNN detect
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(img_rgb)

                if len(results) == 0:
                    continue

                # ناخد أول face
                x, y, w, h = results[0]['box']

                face = img[y:y+h, x:x+w]

                # preprocessing
                face = preprocess_image(face)

                # RGB for encoding
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(face_rgb)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)

print("Encodings done with MTCNN ✅")

# حفظ
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved successfully ✅")