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
│   ├── requirements.txt
│   ├── Procfile                  (Render deployment)
│   └── render.yaml               (Render config)
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── vercel.json               (Vercel config)
└── README.md
```

## How to Run (Local)

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model

```bash
python model/train_model.py
```

### 3. Start the Backend Server

```bash
python backend/app.py
```

Flask starts at `http://127.0.0.1:5000`.

### 4. Open the Frontend

Open `frontend/index.html` in your browser. Fill in the soil metrics and click **Recommend Crop**.

---

## Deploy to Render (Backend API)

1. Push the project to a GitHub repository.
2. Log in to [Render](https://render.com) and click **New + > Web Service**.
3. Connect your GitHub repo and set:
   - **Root Directory:** `backend`
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers=2 --bind=0.0.0.0:$PORT --timeout=120`
4. Click **Create Web Service**.
5. Once deployed, copy your Render URL (e.g. `https://crop-api.onrender.com`).

> Make sure `crop_model.pkl` and `crop_scaler.pkl` are committed to the repo (in the `model/` folder).

## Deploy to Vercel (Frontend)

1. Push the project to a GitHub repository.
2. Log in to [Vercel](https://vercel.com) and click **Add New > Project**.
3. Import your GitHub repo and set:
   - **Root Directory:** `frontend`
   - **Framework Preset:** `Other`
4. Click **Deploy**.
5. After deployment, go to your Vercel project **Settings > Environment Variables** and add:
   - **Key:** `__API_URL__`
   - **Value:** `https://crop-api.onrender.com` (your Render URL)
6. Go to **Deployments**, find the latest deployment, click **... > Redeploy**.
7. Your frontend is now live at `https://your-project.vercel.app`.

Alternatively, edit `frontend/index.html` and uncomment the `__API_URL__` line with your Render URL, then commit and redeploy.

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
- **Backend:** Flask, gunicorn, flask-cors
- **Frontend:** HTML, CSS, JavaScript (Fetch API)
- **Hosting:** Render (backend), Vercel (frontend)
