from flask import Flask, render_template, request
import cv2
import numpy as np
import mediapipe as mp
import joblib
import sys
import os

# Import utils
sys.path.append("..")
from utils import extract_landmarks, normalize_landmarks

app = Flask(__name__)

# ------------------------------
# LOAD TRAINED MODEL
# ------------------------------
MODEL_PATH = "models/hand_verifier.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Run train.py first.")

loaded_model = joblib.load(MODEL_PATH)    # {'clf': clf, 'scaler': scaler}
clf = loaded_model["clf"]
scaler = loaded_model["scaler"]

# Mediapipe Hands
mp_hands = mp.solutions.hands


# ------------------------------
# ACCESS VERIFICATION LOGIC
# ------------------------------
def verify_access(norm_data):
    """Matches same logic you placed inside train.py"""
    pred_prob = clf.predict_proba(norm_data)[0]
    pred_label = clf.classes_[pred_prob.argmax()]
    confidence = float(pred_prob.max())

    if pred_label == "authorized" and confidence > 0.80:
        return {"status": "access_granted", "confidence": confidence}
    else:
        return {"status": "access_denied", "confidence": confidence}


# ------------------------------
# MAIN PAGE
# ------------------------------
@app.route("/")
def login_page():
    return render_template("login.html")


# ------------------------------
# VERIFY HAND API
# ------------------------------
@app.route("/verify", methods=["POST"])
def verify_hand():
    """Receive webcam → extract hand → preprocess → model → return result"""

    file = request.files.get("frame")
    if not file:
        return {"status": "error", "message": "No frame received"}

    # Convert to OpenCV image
    frame_data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

    # Run mediapipe
    with mp_hands.Hands(static_image_mode=True) as hands:
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not result.multi_hand_landmarks:
            return {"status": "nohand"}

        # Feature extraction
        landmarks = extract_landmarks(result.multi_hand_landmarks[0])

        # Normalize (scale)
        norm = scaler.transform([landmarks])

        # Predict using same logic from train.py
        access_result = verify_access(norm)

        return access_result


# ------------------------------
# PAGE ROUTES
# ------------------------------
@app.route("/success")
def success():
    return render_template("success.html")

@app.route("/denied")
def denied():
    return render_template("denied.html")


# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
