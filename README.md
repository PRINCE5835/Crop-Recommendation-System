# Crop Recommendation System based on Soil Data

A beginner-friendly full-stack ML project that recommends the best crop to grow based on soil nutrient and environmental data.

## Dataset

**Kaggle Dataset:** [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

1. Download the CSV from the link above.
2. Rename it to `Crop_recommendation.csv` (if needed).
3. Place it inside the `dataset/` folder.

## Project Structure

```
├── dataset/
│   └── Crop_recommendation.csv   (you download this)
├── model/
│   ├── train_model.py            (trains the KNN model)
│   ├── crop_model.pkl            (generated after training)
│   └── crop_scaler.pkl           (generated after training)
├── backend/
│   ├── app.py                    (Flask API server)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## How to Run

### 1. Install Dependencies

Open a terminal in the project root folder and run:

```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model

```bash
python model/train_model.py
```

This will:
- Load the dataset
- Scale the features using StandardScaler
- Train a KNeighborsClassifier (k=5)
- Save `crop_model.pkl` and `crop_scaler.pkl` inside the `model/` folder

### 3. Start the Backend Server

```bash
python backend/app.py
```

The Flask server will start at `http://127.0.0.1:5000`.

### 4. Open the Frontend

Open `frontend/index.html` in your browser. Fill in the soil metrics and click **Recommend Crop**.

## API Endpoint

### POST /predict

**Request body** (JSON):
```json
{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25.6,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.0
}
```

**Response** (JSON):
```json
{
    "recommended_crop": "rice",
    "top_3_crops": [
        { "crop": "rice", "confidence": 100.0 },
        { "crop": "mothbeans", "confidence": 0.0 },
        { "crop": "blackgram", "confidence": 0.0 }
    ]
}
```

## Tech Stack

- **ML:** Python, scikit-learn (KNN), pandas, numpy, joblib
- **Backend:** Flask, flask-cors
- **Frontend:** HTML, CSS, JavaScript (Fetch API)
