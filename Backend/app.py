"""
app.py — Flask REST API for AI Fraud Detection System
Run: python app.py
"""

import os, json, sqlite3, datetime, hashlib, secrets
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import joblib, numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend on any origin during development

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH    = "database.db"
MODEL_PATH = "model.pkl"
META_PATH  = "model_meta.json"

FEATURES = [
    "amount", "hour", "day_of_week", "distance_from_home",
    "merchant_risk", "prev_txn_gap_hrs", "is_online",
    "num_daily_txns", "velocity_score",
]

# ── Model loading ────────────────────────────────────────────────────────────
model = None
model_meta = {}

def load_model():
    global model, model_meta
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                model_meta = json.load(f)
        print("✅  Model loaded.")
    else:
        print("⚠️   model.pkl not found — run train_model.py first.")

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id          TEXT    UNIQUE NOT NULL,
            timestamp       TEXT    NOT NULL,
            card_number     TEXT    NOT NULL,
            merchant_name   TEXT    NOT NULL,
            amount          REAL    NOT NULL,
            hour            INTEGER NOT NULL,
            day_of_week     INTEGER NOT NULL,
            distance_from_home REAL NOT NULL,
            merchant_risk   REAL    NOT NULL,
            prev_txn_gap_hrs REAL   NOT NULL,
            is_online       INTEGER NOT NULL,
            num_daily_txns  INTEGER NOT NULL,
            velocity_score  REAL    NOT NULL,
            fraud_probability REAL  NOT NULL,
            is_fraud        INTEGER NOT NULL,
            status          TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id      TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            severity    TEXT NOT NULL,
            message     TEXT NOT NULL,
            resolved    INTEGER DEFAULT 0
        );
        """)
        db.commit()
        db.close()
        print("✅  Database initialised.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def mask_card(card):
    c = card.replace(" ", "").replace("-", "")
    return "**** **** **** " + c[-4:] if len(c) >= 4 else "****"

def risk_label(prob):
    if prob >= 0.75: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"

def explain_fraud(features, prob):
    reasons = []
    if features["amount"] > 1500:
        reasons.append(f"Unusually high transaction amount (${features['amount']:.2f})")
    if features["hour"] in list(range(0, 5)):
        reasons.append(f"Transaction at unusual hour ({features['hour']}:00)")
    if features["distance_from_home"] > 50:
        reasons.append(f"Far from home location ({features['distance_from_home']:.1f} km)")
    if features["merchant_risk"] > 0.7:
        reasons.append(f"High-risk merchant category (score {features['merchant_risk']:.2f})")
    if features["velocity_score"] > 0.7:
        reasons.append("High transaction velocity detected")
    if features["num_daily_txns"] > 8:
        reasons.append(f"Abnormal number of daily transactions ({features['num_daily_txns']})")
    if features["prev_txn_gap_hrs"] < 0.5:
        reasons.append("Very short gap since last transaction")
    if not reasons:
        reasons.append("Pattern anomaly detected by ML model")
    return reasons

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_meta": model_meta,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Main fraud prediction endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)

    # --- Validate required fields ---
    required = ["card_number", "merchant_name", "amount"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # --- Derive features ---
    now = datetime.datetime.utcnow()
    features = {
        "amount":              float(data.get("amount", 0)),
        "hour":                int(data.get("hour", now.hour)),
        "day_of_week":         int(data.get("day_of_week", now.weekday())),
        "distance_from_home":  float(data.get("distance_from_home", 5)),
        "merchant_risk":       float(data.get("merchant_risk", 0.2)),
        "prev_txn_gap_hrs":    float(data.get("prev_txn_gap_hrs", 24)),
        "is_online":           int(data.get("is_online", 0)),
        "num_daily_txns":      int(data.get("num_daily_txns", 2)),
        "velocity_score":      float(data.get("velocity_score", 0.1)),
    }

    X = np.array([[features[f] for f in FEATURES]])
    prob = float(model.predict_proba(X)[0][1])
    is_fraud = int(prob >= 0.5)

    # --- Persist to DB ---
    db = get_db()
    txn_id = "TXN-" + secrets.token_hex(6).upper()
    ts     = now.isoformat()

    db.execute("""
        INSERT INTO transactions
        (txn_id, timestamp, card_number, merchant_name, amount, hour, day_of_week,
         distance_from_home, merchant_risk, prev_txn_gap_hrs, is_online,
         num_daily_txns, velocity_score, fraud_probability, is_fraud, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        txn_id, ts,
        mask_card(data["card_number"]),
        data["merchant_name"],
        features["amount"], features["hour"], features["day_of_week"],
        features["distance_from_home"], features["merchant_risk"],
        features["prev_txn_gap_hrs"], features["is_online"],
        features["num_daily_txns"], features["velocity_score"],
        prob, is_fraud,
        "BLOCKED" if is_fraud else "APPROVED",
    ))

    if is_fraud or prob >= 0.4:
        severity = "HIGH" if prob >= 0.75 else "MEDIUM"
        db.execute("""
            INSERT INTO alerts (txn_id, timestamp, severity, message)
            VALUES (?,?,?,?)
        """, (txn_id, ts, severity,
              f"Suspicious transaction {txn_id}: prob={prob:.2%}"))

    db.commit()

    return jsonify({
        "txn_id":           txn_id,
        "timestamp":        ts,
        "is_fraud":         bool(is_fraud),
        "fraud_probability": round(prob, 4),
        "risk_level":       risk_label(prob),
        "status":           "BLOCKED" if is_fraud else "APPROVED",
        "card_masked":      mask_card(data["card_number"]),
        "merchant":         data["merchant_name"],
        "amount":           features["amount"],
        "fraud_reasons":    explain_fraud(features, prob) if is_fraud or prob >= 0.4 else [],
    })


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Return paginated transaction history."""
    db     = get_db()
    limit  = min(int(request.args.get("limit", 20)), 100)
    offset = int(request.args.get("offset", 0))
    filter_fraud = request.args.get("fraud")

    query  = "SELECT * FROM transactions"
    params = []
    if filter_fraud in ("0", "1"):
        query += " WHERE is_fraud = ?"
        params.append(int(filter_fraud))
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    rows = db.execute(query, params).fetchall()
    total = db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    return jsonify({
        "total": total,
        "rows":  [dict(r) for r in rows],
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Dashboard statistics."""
    db = get_db()
    total     = db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    frauds    = db.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud=1").fetchone()[0]
    today_str = datetime.datetime.utcnow().date().isoformat()
    today_txn = db.execute(
        "SELECT COUNT(*) FROM transactions WHERE timestamp LIKE ?",
        (today_str + "%",)
    ).fetchone()[0]
    avg_amount = db.execute(
        "SELECT AVG(amount) FROM transactions"
    ).fetchone()[0] or 0

    alerts_open = db.execute(
        "SELECT COUNT(*) FROM alerts WHERE resolved=0"
    ).fetchone()[0]

    # fraud by hour (last 50 txns)
    hourly = db.execute("""
        SELECT hour, COUNT(*) as cnt, SUM(is_fraud) as fraud_cnt
        FROM (SELECT * FROM transactions ORDER BY id DESC LIMIT 200)
        GROUP BY hour ORDER BY hour
    """).fetchall()

    # recent fraud txns
    recent_fraud = db.execute("""
        SELECT txn_id, timestamp, amount, merchant_name, fraud_probability
        FROM transactions WHERE is_fraud=1
        ORDER BY id DESC LIMIT 5
    """).fetchall()

    return jsonify({
        "total_transactions": total,
        "total_fraud":        frauds,
        "fraud_rate":         round((frauds / total * 100) if total else 0, 2),
        "today_transactions": today_txn,
        "avg_amount":         round(avg_amount, 2),
        "open_alerts":        alerts_open,
        "hourly_distribution": [dict(r) for r in hourly],
        "recent_frauds":      [dict(r) for r in recent_fraud],
        "model_accuracy":     model_meta.get("accuracy", "N/A"),
        "model_auc":          model_meta.get("roc_auc", "N/A"),
    })


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    db = get_db()
    db.execute("UPDATE alerts SET resolved=1 WHERE id=?", (alert_id,))
    db.commit()
    return jsonify({"success": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_model()
    print("\n🚀  Fraud Detection API running on http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
