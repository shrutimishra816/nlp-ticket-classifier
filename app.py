"""
app.py — NLP Ticket Classifier API + Frontend
----------------------------------------------
Serves the React frontend and a /classify endpoint.
Auto-trains the model on first startup if no joblib found.

Endpoints:
  GET  /           → React frontend
  POST /classify   → { text } → { category, priority, confidence, all_scores }
  POST /classify/batch → [{ id, text }] → list of results
  GET  /health     → service status
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD = os.path.join(BASE_DIR, "frontend", "build")
MODEL_PATH = os.path.join(BASE_DIR, "ticket_classifier.joblib")

app = Flask(__name__, static_folder=FRONTEND_BUILD)
CORS(app)

PRIORITY_MAP = {
    "billing":          "High",
    "technical_issue":  "High",
    "account_access":   "High",
    "refund":           "High",
    "data_privacy":     "Critical",
    "performance":      "Medium",
    "feature_request":  "Low",
    "onboarding":       "Low",
}

CATEGORY_LABELS = {
    "billing":          "Billing",
    "technical_issue":  "Technical Issue",
    "account_access":   "Account Access",
    "feature_request":  "Feature Request",
    "refund":           "Refund",
    "onboarding":       "Onboarding",
    "performance":      "Performance",
    "data_privacy":     "Data Privacy",
}

# ── Auto-train if model missing ───────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("[INFO] No model found — training now...")
    import sys
    sys.path.insert(0, BASE_DIR)
    from classifier import generate_dataset, train_and_evaluate
    df = generate_dataset(n_per_class=200)
    train_and_evaluate(df)
    print("[INFO] Training complete.")

model = joblib.load(MODEL_PATH)
CLASSES = list(model.classes_)
print(f"[INFO] Model loaded. Classes: {CLASSES}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "ticket_classifier", "categories": len(CLASSES)})


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text field is required"}), 400

    proba = model.predict_proba([text])[0]
    top_idx = int(np.argmax(proba))
    category = CLASSES[top_idx]
    confidence = float(proba[top_idx])

    all_scores = [
        {"category": CLASSES[i], "label": CATEGORY_LABELS[CLASSES[i]], "score": round(float(proba[i]), 4)}
        for i in np.argsort(proba)[::-1]
    ]

    return jsonify({
        "category": category,
        "label": CATEGORY_LABELS[category],
        "priority": PRIORITY_MAP[category],
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    })


@app.route("/classify/batch", methods=["POST"])
def classify_batch():
    data = request.get_json(force=True)
    tickets = data.get("tickets", [])
    if not tickets:
        return jsonify({"error": "tickets array is required"}), 400

    texts = [t.get("text", "") for t in tickets]
    probas = model.predict_proba(texts)
    results = []

    for i, (ticket, proba) in enumerate(zip(tickets, probas)):
        top_idx = int(np.argmax(proba))
        category = CLASSES[top_idx]
        results.append({
            "id": ticket.get("id", f"TKT-{i+1:03d}"),
            "text": ticket.get("text", ""),
            "category": category,
            "label": CATEGORY_LABELS[category],
            "priority": PRIORITY_MAP[category],
            "confidence": round(float(proba[top_idx]), 4),
        })

    return jsonify({"results": results})


# ── Serve React ───────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path and os.path.exists(os.path.join(FRONTEND_BUILD, path)):
        return send_from_directory(FRONTEND_BUILD, path)
    return send_from_directory(FRONTEND_BUILD, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n TicketSight API running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
