# CMI — Detect Behavior with Sensor Data

**Repository:** `S2V3/CMI-Detect-Behaviour-using-sensor-data`  
**Competition referenced:** [CMI - Detect Behavior with Sensor Data (Child Mind Institute)](https://www.kaggle.com/competitions/cmi-detect-behavior-using-sensor-data)

---

##  Project Overview

This project contains code and experiments for the Kaggle competition **"CMI - Detect Behavior with Sensor Data"**.  
The task is to build models that classify wrist-worn sensor recordings into **body-focused repetitive behaviors (BFRB-like gestures)** and **non-BFRB-like gestures**, using IMU/time-series data from a wrist device.

The dataset includes multimodal sensor streams (accelerometer, gyroscope, etc.) and per-gesture labels.

---

##  Repository Structure

```
.
├── data/                   # (not included) place dataset files here or mount Kaggle data
├── notebooks/              # EDA and baseline notebooks
├── src/                    # Training, model, and data-preprocessing code
├── models/                 # Saved model weights (optional)
├── submissions/            # Generated submission CSVs
├── requirements.txt        # Python dependencies
├── config.py                # Project configuration (example)
└── README.md                # This file
```


##  Getting the data

1. Go to the [Kaggle competition page](https://www.kaggle.com/competitions/cmi-detect-behavior-using-sensor-data)
2. Accept competition rules
3. Download the training/test data and place it like:

```
data/
 ├── train/
 ├── test/
 ├── train_labels.csv
 └── metadata.csv
```

---

## ️ Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Minimal `requirements.txt` example:

```
numpy
pandas
scipy
scikit-learn
torch>=1.8
tqdm
matplotlib
seaborn
```

---

##  How to run

### 1. Preprocess

```bash
python src/preprocess.py --input_dir data/train --output_dir data/processed --config config.py
```

### 2. Train

```bash
python src/train.py --data_dir data/processed --model_dir models/exp1 --epochs 50 --batch-size 64
```

### 3. Inference

```bash
python src/inference.py --model_path models/exp1/best.pth --test_dir data/test --output submissions/submission_exp1.csv
```

---

##  Notebooks

Open `notebooks/` for:

- Sensor data EDA
- Visualization (gesture plots, motion trajectories)
- Baseline models: 1D CNN, LSTM, Transformer

---

##  Tips

- Handle missing timestamps / varying sampling rates carefully
- Try multiple window lengths (1s, 2s, 3s)
- Use class weights or focal loss for class imbalance
- Augmentation: noise, scaling, time warping

---

##  References

- [CMI - Detect Behavior with Sensor Data (Kaggle)](https://www.kaggle.com/competitions/cmi-detect-behavior-using-sensor-data)

---
