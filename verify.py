# verify.py
import cv2, mediapipe as mp, joblib, argparse
from utils import landmarks_to_vector, normalize_vector, ensure_data_dirs
import numpy as np, os

ensure_data_dirs()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = os.path.join('models','hand_verifier.joblib')

def verify_live(target_label=None):
    if not os.path.exists(MODEL_PATH):
        print('No trained model found. Run train.py first.')
        return
    data = joblib.load(MODEL_PATH)
    clf = data['clf']; scaler = data['scaler']
    cap = cv2.VideoCapture(0)
    print('Starting live verification. Press q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(image)
        text = 'No hand'
        if res.multi_hand_landmarks:
            vec = landmarks_to_vector(res.multi_hand_landmarks[0])
            vec = normalize_vector(vec)
            v_s = scaler.transform(vec.reshape(1,-1))
            probas = clf.predict_proba(v_s)[0]
            pred = clf.predict(v_s)[0]
            score = probas.max()
            text = f'Pred: {pred} ({score:.2f})'
            if target_label is not None:
                accept = (pred == target_label and score>0.5)
                text += ' => ' + ('ACCEPT' if accept else 'REJECT')
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        if res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Verify', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['verify'])
    parser.add_argument('label', nargs='?', default=None)
    args = parser.parse_args()
    if args.cmd == 'verify':
        verify_live(args.label)
