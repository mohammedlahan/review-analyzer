# Customer Review Analyzer

A beginner-friendly web app that classifies customer reviews as **positive**, **neutral**, or **negative** using a fine-tuned RoBERTa model.

**Stack:** FastAPI (Python) · plain HTML/CSS/JS · Chart.js

---

## Project Structure

```
review-analyzer/
├── backend/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app + all endpoints
│   └── model/           ← place your 5 model files here (see below)
│       └── .gitkeep
├── frontend/
│   └── index.html       ← entire UI in one file
├── sample_reviews.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/mohammedlahan/review-analyzer.git
cd review-analyzer
```

### 2. Add your model files

Copy these 5 files into `backend/model/`:

| File | Description |
|---|---|
| `model.safetensors` | Trained model weights |
| `config.json` | Model architecture config |
| `tokenizer.json` | Tokeniser vocabulary |
| `tokenizer_config.json` | Tokeniser settings |
| `training_args.bin` | Training hyper-parameters |

> **Note:** The `.safetensors` and `.bin` files are excluded from Git (see `.gitignore`) because they are large.
> See **"How to access the model"** below for sharing options.

### 3. Create virtual environment & install
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the app
Go to → **http://localhost:8000**

---

## API Reference

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/api/health` | — | `{"status":"ok"}` |
| POST | `/api/analyze` | `{"text":"…"}` | `{sentiment, confidence, scores}` |
| POST | `/api/batch` | CSV file (multipart) | `{results[], summary}` |

---

## How to Access / Share the Model

The model weights are **not stored in this Git repo** (they can be hundreds of MB).
Here are three ways to distribute them:

### Option A — Hugging Face Hub (recommended)
1. Create a free account at https://huggingface.co
2. Create a new model repository (can be private)
3. Upload all 5 model files
4. In `backend/main.py` change:
   ```python
   # Instead of local path:
   classifier = pipeline("text-classification", model="backend/model", top_k=None)

   # Use your HF repo ID:
   classifier = pipeline("text-classification", model="your-username/your-model-name", top_k=None)
   ```
   Now anyone who clones the repo just needs to run the server — the model downloads automatically.

### Option B — Google Drive / OneDrive
1. Upload the `model/` folder as a zip to Google Drive
2. Share the link in your README
3. Cloners download and unzip into `backend/model/`

### Option C — Git LFS (for private/small teams)
```bash
git lfs install
git lfs track "*.safetensors" "*.bin"
git add .gitattributes
git add backend/model/
git commit -m "Add model via LFS"
git push
```
GitHub free tier gives 1 GB of LFS storage.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `OSError: Can't load model` | Make sure all 5 files are in `backend/model/` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| Port 8000 in use | Use `--port 8001` and open `http://localhost:8001` |
| Slow first request | Normal — PyTorch JIT warms up on first call |
