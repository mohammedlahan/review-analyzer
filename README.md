# Customer Review Analyzer

A beginner-friendly web app that classifies customer reviews as **positive**, **neutral**,
or **negative** using a fine-tuned RoBERTa model.

**Stack:** FastAPI (Python) · plain HTML/CSS/JS · Chart.js  
**Model:** 🤗 [mohammedlahan/review-sentiment-model](https://huggingface.co/mohammedlahan/review-sentiment-model)

---

## ✨ Features

- **Single review** — type any review, get sentiment + confidence bars
- **Batch CSV** — upload up to 500 reviews, get a full dashboard with chart
- **Mobile-friendly** — works on phones and tablets

---

## 🚀 Quick Start (3 steps)

### Step 1 — Clone the repo
```bash
git clone https://github.com/mohammedlahan/review-analyzer.git
cd review-analyzer
```

### Step 2 — Create virtual environment & install
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Run the server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open → **http://localhost:8000**

> **First run:** The model downloads automatically from Hugging Face (~500 MB).
> This takes a few minutes once. After that it is cached and loads instantly.

---

## 🧠 Model

The sentiment model is hosted on Hugging Face Hub:

🔗 **https://huggingface.co/mohammedlahan/review-sentiment-model**

| Detail | Value |
|---|---|
| Architecture | RoBERTa-base |
| Task | Text classification (3 labels) |
| Labels | `negative` · `neutral` · `positive` |
| Input limit | 128 tokens (longer text is truncated) |

**You do NOT need to download the model manually.**
The app fetches it automatically on first run using the `transformers` library.

---

## 📁 Project Structure

```
review-analyzer/
├── backend/
│   ├── __init__.py        ← makes backend/ a Python package
│   ├── main.py            ← FastAPI app + all API endpoints
│   └── model/             ← (empty) model loads from HuggingFace
├── frontend/
│   └── index.html         ← entire UI in one HTML file
├── sample_reviews.csv     ← 20 test reviews for batch upload
├── requirements.txt       ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🔌 API Reference

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/api/health` | — | `{"status":"ok"}` |
| POST | `/api/analyze` | `{"text":"your review"}` | `{sentiment, confidence, scores}` |
| POST | `/api/batch` | CSV file (multipart) | `{results[], summary}` |

**Example:**
```bash
curl -X POST http://localhost:8000/api/analyze \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Amazing product, very happy!\"}"
```
```json
{
  "sentiment": "positive",
  "confidence": 0.96,
  "scores": {"negative": 0.01, "neutral": 0.03, "positive": 0.96}
}
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| Slow first startup | Normal — model is downloading from HuggingFace (~500 MB) |
| Port 8000 in use | Use `--port 8001` and open `http://localhost:8001` |
| `OSError: Can't load model` | Check your internet connection (needed for first download) |

---

## 📊 Model Limitations

- **Neutral is the hardest class** — mixed reviews often get mis-classified
- **Sarcasm is missed** — "Oh great, another broken charger" may read as positive  
- **128-token cap** — very long reviews are silently truncated
