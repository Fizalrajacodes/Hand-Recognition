# calibrate.py
# Simple threshold suggestion using classifier scores
import pandas as pd, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from utils import ensure_data_dirs

ensure_data_dirs()
DATA_CSV = os.path.join('data','hand_landmarks.csv')
MODEL_PATH = os.path.join('models','hand_verifier.joblib')

def suggest_threshold():
    if not os.path.exists(DATA_CSV):
        print('Need captured data to calibrate.')
        return
    df = pd.read_csv(DATA_CSV)
    data = joblib.load(MODEL_PATH)
    clf = data['clf']; scaler = data['scaler']
    X = df[[c for c in df.columns if c!='label']].values
    y = df['label'].values
    Xs = scaler.transform(X)
    probas = clf.predict_proba(Xs)
    scores = probas.max(axis=1)
    # basic impostor vs genuine: below median score treat as impostor
    importances = {}
    print('Score stats:', np.min(scores), np.mean(scores), np.median(scores), np.max(scores))
    # Save histogram plot
    plt.figure()
    plt.hist(scores, bins=30)
    plt.title('Classifier max-probability scores')
    plt.xlabel('score')
    plt.ylabel('count')
    plt.savefig('models/score_hist.png')
    print('Saved models/score_hist.png')
    print('Suggested threshold (median):', np.median(scores))
if __name__ == '__main__':
    suggest_threshold()
