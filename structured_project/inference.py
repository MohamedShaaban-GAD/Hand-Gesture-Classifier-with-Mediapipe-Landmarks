import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import MODEL_OUTPUT_PATH


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FEATURE_COLS = [
    f"{axis}{i}" for i in range(1, 22) for axis in ("x", "y", "z")
]


def _init_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(frame, hands):
    """Return a 63-element numpy array of hand landmarks, or None."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None


def run_webcam(model_path: str = MODEL_OUTPUT_PATH, encoder: LabelEncoder | None = None):
    """Run real-time gesture recognition from webcam."""
    model = joblib.load(model_path)
    hands = _init_hands()

    cap = cv2.VideoCapture(0)
    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame, hands)

        if landmarks is not None:
            landmarks_df = pd.DataFrame(
                landmarks.reshape(1, -1), columns=FEATURE_COLS,
            )
            pred_encoded = model.predict(landmarks_df)[0]

            if encoder is not None:
                label = encoder.inverse_transform([pred_encoded])[0]
            else:
                label = str(pred_encoded)

            cv2.putText(
                frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
            )

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
