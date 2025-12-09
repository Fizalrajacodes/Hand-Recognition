# capture.py
# Usage: python capture.py capture <label> --n 30
import cv2, mediapipe as mp, time, argparse
from utils import landmarks_to_vector, save_rows, ensure_data_dirs, normalize_vector

ensure_data_dirs()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

def capture_samples(user_label, num_samples=30, delay=0.1):
    cap = cv2.VideoCapture(0)
    collected = 0
    rows = []
    print(f"Start capturing {num_samples} samples for user='{user_label}'")
    time.sleep(1.0)
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            vec = landmarks_to_vector(lm)
            vec = normalize_vector(vec)
            rows.append([user_label] + vec.tolist())
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            collected += 1
            cv2.putText(frame, f"Collected: {collected}/{num_samples}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture (press q to quit)", frame)
        if cv2.waitKey(int(delay*1000)) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()
    save_rows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['capture'])
    parser.add_argument('label')
    parser.add_argument('--n', type=int, default=30)
    args = parser.parse_args()
    if args.cmd == 'capture':
        capture_samples(args.label, num_samples=args.n)
