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

# NLP Ticket Classifier → Airtable Ops Dashboard

> **60% faster triage** on 10,000+ support tickets · **$30K projected annual saving** · 8-category multi-class classifier with Airtable priority dashboard

---

## What This Does

Classifies incoming support tickets into 8 categories, tags each with a priority level (Critical / High / Medium / Low), and exports an Airtable-ready CSV — so ops teams can act on tickets without touching any ML output directly.

**Categories:** Billing · Technical Issue · Account Access · Feature Request · Refund · Onboarding · Performance · Data Privacy

---

## Architecture

```
Raw Support Tickets (CSV / CRM export)
        │
        ▼
  TF-IDF Vectoriser (bigrams, 15k features)
        │
        ▼
  Logistic Regression Classifier
  (BERT fine-tuning for production — see notebook)
        │
        ▼
  Category + Priority Tag + Confidence Score
        │
        ▼
  airtable_export.csv → Ops Team Dashboard
  (no ML knowledge required to use)
```

---

## Repo Structure

```
nlp-ticket-classifier/
├── classifier.py                    # Full training + prediction + Airtable export
├── ticket_classifier_notebook.ipynb # Step-by-step walkthrough with impact analysis
├── evaluation_report.json           # Per-category precision, recall, F1 (auto-generated)
├── airtable_export.csv              # Sample output — paste directly into Airtable
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train model + classify sample tickets + export Airtable CSV
python classifier.py --full

# Train only
python classifier.py --train

# Classify from existing model
python classifier.py --predict
```

---

## Sample Output

| Ticket ID | Category | Priority | Confidence |
|-----------|----------|----------|------------|
| TKT-001 | Billing | High | 92% |
| TKT-002 | Technical Issue | High | 68% |
| TKT-003 | Account Access | High | 91% |
| TKT-004 | Feature Request | Low | 78% |
| TKT-005 | Data Privacy | **Critical** | 94% |
| TKT-006 | Performance | Medium | 73% |
| TKT-007 | Onboarding | Low | 87% |
| TKT-008 | Refund | High | 93% |

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

## Model Details

| Parameter | Value |
|-----------|-------|
| Vectoriser | TF-IDF, bigrams, 15k features |
| Classifier | Logistic Regression (C=5, lbfgs) |
| Production | BERT fine-tuned — 89% accuracy (see notebook) |
| Categories | 8 |
| Evaluation | Precision · Recall · F1 · Confusion Matrix per category |

Full evaluation report auto-saved to `evaluation_report.json` on training.

---

## Tech Stack
`Python` · `Scikit-Learn` · `TF-IDF` · `BERT / HuggingFace Transformers` · `Airtable` · `Pandas`
