# 🛡️ ShieldAI — AI-Based Smart Fraud Detection & Secure Payment System

> Capstone Project | Machine Learning + Flask REST API + Vanilla JS Dashboard

---

## 📁 Project Structure

```
fraud-detection-system/
├── backend/
│   ├── app.py            ← Flask REST API (5 endpoints)
│   ├── train_model.py    ← ML model training script
│   ├── model.pkl         ← Trained RandomForest model (auto-generated)
│   ├── model_meta.json   ← Model accuracy/AUC metadata (auto-generated)
│   ├── database.db       ← SQLite database (auto-generated)
│   └── requirements.txt  ← Python dependencies
│
├── frontend/
│   ├── index.html        ← Single-page application shell
│   ├── style.css         ← Dark industrial UI design
│   └── script.js         ← Full SPA logic (dashboard, predict, tables, alerts)
│
└── README.md
```

---

## ⚙️ Setup & Run

### Step 1 — Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Train the ML model (run once)
```bash
python train_model.py
```
This generates `model.pkl` and `model_meta.json`.  
Expected output: ~97%+ accuracy, ~0.99 ROC-AUC on synthetic data.

### Step 3 — Start the Flask backend
```bash
python app.py
```
API runs on **http://localhost:5000**

### Step 4 — Open the frontend
Open `frontend/index.html` directly in your browser.  
(No build step needed — pure HTML/CSS/JS)

---

## 🌐 API Endpoints

| Method | Endpoint                        | Description                    |
|--------|---------------------------------|--------------------------------|
| GET    | `/api/health`                   | Health check + model status    |
| POST   | `/api/predict`                  | Run fraud prediction           |
| GET    | `/api/transactions`             | Paginated transaction history  |
| GET    | `/api/stats`                    | Dashboard statistics           |
| GET    | `/api/alerts`                   | Security alerts list           |
| POST   | `/api/alerts/<id>/resolve`      | Resolve an alert               |

### POST `/api/predict` — Request Body
```json
{
  "card_number": "4111 1111 1111 1234",
  "merchant_name": "Amazon",
  "amount": 149.99,
  "hour": 14,
  "distance_from_home": 5.0,
  "merchant_risk": 0.15,
  "prev_txn_gap_hrs": 18.0,
  "is_online": 1,
  "num_daily_txns": 3,
  "velocity_score": 0.1
}
```

### POST `/api/predict` — Response
```json
{
  "txn_id": "TXN-A1B2C3",
  "is_fraud": false,
  "fraud_probability": 0.032,
  "risk_level": "LOW",
  "status": "APPROVED",
  "card_masked": "**** **** **** 1234",
  "fraud_reasons": []
}
```

---

## 🤖 ML Model

| Property       | Value                         |
|----------------|-------------------------------|
| Algorithm      | Random Forest Classifier      |
| Training data  | 10,000 synthetic transactions |
| Fraud rate     | ~3% (realistic class balance) |
| Features       | 9 behavioral + transaction    |
| Preprocessing  | StandardScaler pipeline       |
| Class weights  | Balanced (handles imbalance)  |

### Features Used
- `amount` — Transaction amount
- `hour` — Hour of day (0–23)
- `day_of_week` — Day (0=Mon, 6=Sun)
- `distance_from_home` — km from cardholder's home
- `merchant_risk` — Merchant category risk score (0–1)
- `prev_txn_gap_hrs` — Hours since last transaction
- `is_online` — Boolean: online vs in-person
- `num_daily_txns` — Number of transactions today
- `velocity_score` — Transaction velocity anomaly score

---

## 🎨 Frontend Features

- **Dashboard** — KPI cards, hourly activity bar chart, recent fraud table
- **Scan Transaction** — Full form with AI risk analysis, animated gauge, fraud reasons
- **Transactions** — Paginated history with status badges and risk indicators
- **Alerts** — Live alerts with HIGH/MEDIUM severity, resolve functionality
- Sample buttons: "Load Legit Sample" and "Load Fraud Sample" for demo

---

## 🔧 Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| ML        | scikit-learn (RandomForest)         |
| Backend   | Python 3.10+, Flask, Flask-CORS     |
| Database  | SQLite3 (built-in)                  |
| Frontend  | HTML5, CSS3, Vanilla JavaScript     |
| Charts    | Chart.js 4 (CDN)                    |
| Fonts     | Google Fonts (Syne + Space Mono)    |

---

## 📊 Database Schema

### `transactions`
Stores every scanned transaction with features, prediction result, and status.

### `alerts`
Auto-generated for any transaction with fraud_probability ≥ 0.40, with HIGH/MEDIUM severity and resolve tracking.
