import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download

# download model from Hugging Face
model_path = hf_hub_download(
    repo_id="emann123/face-reg-model",   #name of repository
    filename="best_emotion_model.keras"
)
model = load_model(model_path)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

mp_face_detection = mp.solutions.face_detection

def detect_and_predict_emotion(frame):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [{"emotion": "No face detected", "confidence": 0.0}], frame

        detections = []
        h, w, _ = frame.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            roi = frame[y:y+height, x:x+width]
            roi = cv2.resize(roi, (224, 224))
            roi = img_to_array(roi) / 255.0
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = float(np.max(preds))
            detections.append({"emotion": emotion, "confidence": confidence})

            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return detections, frame
