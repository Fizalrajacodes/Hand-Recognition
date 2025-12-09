# Hand Recognition — Landmark-based Verifier (Prototype)

**What this is**
- A complete, ready-to-run prototype for a hand/finger landmark-based biometric verifier using a webcam.
- Uses MediaPipe for 21 hand landmarks, stores landmark vectors per-user, trains a lightweight SVM verifier, and performs live verification.
- Includes simple liveness (motion) check and threshold calibration script.

**Contents**
- `capture.py` — capture landmarks from webcam into `data/hand_landmarks.csv`
- `train.py` — train an SVM verifier and save `models/hand_verifier.joblib`
- `verify.py` — live verification GUI (uses trained model)
- `calibrate.py` — compute thresholds, EER-style curve and suggested threshold
- `utils.py` — shared helper functions (normalization, IO)
- `requirements.txt` — Python packages
- `example_data/sample_hand_landmarks.csv` — tiny example CSV with two users
- `LICENSE` — MIT

**Quick start (Linux / Windows WSL / macOS)**
1. Create virtualenv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Capture samples for user `alice`:
   ```bash
   python capture.py capture alice --n 30
   ```
   Repeat for other users (e.g., `bob`).
3. Train:
   ```bash
   python train.py
   ```
4. Verify live:
   ```bash
   python verify.py verify alice
   ```

**Notes & recommendations**
- Good lighting and consistent hand pose improve results.
- Normalize landmarks (translation, scale, optional rotation) is performed in `utils.py`.
- For production: use secure storage for templates, stronger embeddings (siamese nets), and robust liveness (NIR/depth).

