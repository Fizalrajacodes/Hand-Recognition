# train.py
import pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
from utils import ensure_data_dirs

ensure_data_dirs()
DATA_CSV = os.path.join('data','hand_landmarks.csv')

def train():
    if not os.path.exists(DATA_CSV):
        print('No data found. Run capture first.')
        return
    df = pd.read_csv(DATA_CSV)
    X = df[[c for c in df.columns if c!='label']].values
    y = df['label'].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = SVC(kernel='linear', probability=True).fit(Xs, y)
    joblib.dump({'clf':clf, 'scaler':scaler}, os.path.join('models','hand_verifier.joblib'))
    print('Trained and saved models/hand_verifier.joblib')

if __name__ == '__main__':
    train()
    def verify_access(norm_data, loaded_model):
        pred_prob = loaded_model['clf'].predict_proba(norm_data)[0]
        pred_label =loaded_model['clf'].classes_[pred_prob.argmax()]
        confidence = pred_prob.max()

        # Access logic
        if pred_label == 'authorized' and confidence > 0.8:
            return{"status": "access_granted", "confidence": float(confidence)}
        else:
            return {"status": "access_denied", "confidence": float(confidence)}