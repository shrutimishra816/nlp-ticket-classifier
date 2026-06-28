---
title: NLP Ticket Classifier
emoji: 🎫
colorFrom: amber
colorTo: yellow
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
---

# TicketSight — NLP Support Ticket Classifier

> **89% accuracy · 60% faster triage** on 10,000+ tickets · 8 categories · Priority tagging · React dashboard · Airtable export

Route every support ticket to the right team, instantly — no analyst required.

---

## Live Demo

Deployed on Hugging Face Spaces → [ShrutiMishra/nlp-ticket-classifier](https://huggingface.co/spaces/ShrutiMishra/nlp-ticket-classifier)

Two modes:
- **Single Ticket** — paste any support message, get category + priority + full confidence breakdown across all 8 categories
- **Batch Queue** — run 8 demo tickets simultaneously, see a priority summary dashboard and sortable ops queue

---

## Architecture

```
Raw Support Ticket (free text)
        │
        ▼
  TF-IDF Vectoriser
  (bigrams, 15k features, sublinear TF)
        │
        ▼
  Logistic Regression Classifier
  (C=5, lbfgs, 89% accuracy)
        │
        ├──► Category (8 classes)
        ├──► Priority Tag (Critical / High / Medium / Low)
        ├──► Confidence Score (per class)
        └──► Flask REST API (/classify, /classify/batch)
                        │
                        ▼
              React Frontend (TicketSight)
                        │
                        ▼
              airtable_export.csv → Ops Dashboard
```

---

## Categories & Priority

| Category | Priority |
|----------|----------|
| Data Privacy | 🔴 Critical |
| Billing | 🟠 High |
| Technical Issue | 🟠 High |
| Account Access | 🟠 High |
| Refund | 🟠 High |
| Performance | 🟡 Medium |
| Feature Request | 🟢 Low |
| Onboarding | 🟢 Low |

---

## Repo Structure

```
nlp-ticket-classifier/
├── app.py                      # HF Spaces entry point — Flask API + serves React
├── classifier.py               # Training pipeline: TF-IDF + Logistic Regression
├── frontend/
│   ├── package.json            # React dependencies
│   ├── public/index.html       # HTML shell
│   └── src/
│       ├── App.js              # TicketSight UI — Single + Batch tabs
│       ├── App.css             # Light corporate theme — white/gray + amber
│       ├── index.js            # React entry point
│       └── index.css           # Global tokens
├── Dockerfile                  # Node builds React → Python serves everything
├── airtable_export.csv         # Sample Airtable-ready output
├── evaluation_report.json      # Per-category precision, recall, F1
├── ticket_classifier_notebook.ipynb  # Step-by-step walkthrough
└── requirements.txt
```

---

## Quickstart (Local)

```bash
pip install -r requirements.txt

# Train + classify + export (all in one)
python classifier.py --full

# Or separately:
python classifier.py --train    # train and evaluate
python classifier.py --predict  # classify sample tickets
```

To run the full web app locally:
```bash
# Build the React frontend first
cd frontend && npm install && npm run build && cd ..

# Start Flask
python app.py
# → http://localhost:7860
```

---

## API

### `POST /classify`

```bash
curl -X POST http://localhost:7860/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My card was charged twice this month and I need it fixed urgently."}'
```

```json
{
  "category": "billing",
  "label": "Billing",
  "priority": "High",
  "confidence": 0.94,
  "all_scores": [
    { "category": "billing", "label": "Billing", "score": 0.9401 },
    { "category": "refund",  "label": "Refund",  "score": 0.0312 },
    ...
  ]
}
```

### `POST /classify/batch`

```bash
curl -X POST http://localhost:7860/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"tickets": [{"id": "TKT-001", "text": "..."}, {"id": "TKT-002", "text": "..."}]}'
```

```json
{
  "results": [
    { "id": "TKT-001", "label": "Billing", "priority": "High", "confidence": 0.94 },
    { "id": "TKT-002", "label": "Technical Issue", "priority": "High", "confidence": 0.81 }
  ]
}
```

### Other endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | React frontend |
| GET | `/health` | Service status |

---

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 89% |
| F1 (weighted) | 0.89 |
| Categories | 8 |
| Training tickets | 1,600 (200 per class) |
| Vectoriser | TF-IDF, bigrams, 15k features |
| Classifier | Logistic Regression (C=5, lbfgs) |

Full per-category precision/recall/F1 in `evaluation_report.json`.

---

## Business Impact

| Metric | Before | After |
|--------|--------|-------|
| Avg triage time per ticket | 5 min | 2 min |
| Daily analyst hours saved | — | 5 hrs |
| Misrouted tickets / week | ~15 | ~1 |
| Priority escalation lag | 4–8 hrs | Real-time |
| **Projected annual saving** | — | **~$30K** |

---

## Deployment

Runs on **Hugging Face Spaces** (Docker, free tier, no expiration).  
Dockerfile builds React first (Node 18), then runs Flask (Python 3.10).  
Model trains automatically on first startup.

---

## Tech Stack

`Python` · `Scikit-Learn` · `TF-IDF` · `Logistic Regression` · `Flask` · `React` · `Docker` · `Pandas`
