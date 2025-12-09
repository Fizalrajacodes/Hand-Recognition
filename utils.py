import numpy as np
import pandas as pd
import os

DATA_CSV = os.path.join('data','hand_landmarks.csv')

def ensure_data_dirs():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('example_data', exist_ok=True)

def landmarks_to_vector(landmarks):
    # landmarks: mediapipe LandmarkList
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

def normalize_vector(vec):
    # vec: 63-d (21x3) flattened
    v = vec.reshape(21,3)
    # translate so wrist (0) at origin
    wrist = v[0,:2]
    v[:,:2] = v[:,:2] - wrist
    # scale by palm width (distance between landmarks 0 and 9)
    palm_width = np.linalg.norm(v[0,:2] - v[9,:2]) + 1e-6
    v[:,:2] = v[:,:2] / palm_width
    return v.flatten()

def save_rows(rows, csvpath=DATA_CSV):
    df = pd.DataFrame(rows, columns=['label'] + [f'f{i}' for i in range(63)])
    if os.path.exists(csvpath):
        df_old = pd.read_csv(csvpath)
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(csvpath, index=False)
    print('Saved', csvpath)
def extract_landmarks(landmark_list):
    return landmarks_to_vector(landmark_list)
def normalize_landmarks(landmark_vectors):
    return np.array([normalize_vector(vec) for vec in landmark_vectors])