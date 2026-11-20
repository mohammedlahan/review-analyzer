# PROJECT DESCRIPTION
## Customer Review Analyzer — Complete Technical Documentation

> **Who is this for?** Anyone who wants to understand exactly how this project was
> built — the frontend design, the backend logic, and how the AI model works.
> Written for beginners with clear explanations at every step.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture — The Big Picture](#2-architecture--the-big-picture)
3. [The AI Model — How It Was Created & How It Works](#3-the-ai-model)
4. [The Backend — FastAPI Server](#4-the-backend--fastapi-server)
5. [The Frontend — Building the UI](#5-the-frontend--building-the-ui)
6. [How Everything Connects — Request Lifecycle](#6-how-everything-connects)
7. [Technology Choices Explained](#7-technology-choices-explained)
8. [File-by-File Breakdown](#8-file-by-file-breakdown)

---

## 1. Project Overview

**Customer Review Analyzer** is a full-stack AI web application that reads customer
reviews and classifies them as **positive**, **neutral**, or **negative**.

### What the user can do

| Feature | How it works |
|---|---|
| Type a single review | Gets back a sentiment label + confidence score + 3 coloured bars |
| Upload a CSV file | Processes up to 500 reviews, shows a summary chart and table |
| See top negatives | The 5 most confidently negative reviews are highlighted |
| Read about the model | A dedicated tab explains architecture and limitations |

### Why this project exists

Traditional rule-based systems (checking for words like "bad" or "great") fail on
nuanced text. A transformer model understands context — it knows "not bad" is positive
and "great, another broken charger" is sarcastic.

---

## 2. Architecture — The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   index.html (one file)                    │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │ │
│  │  │  Tab 1       │ │  Tab 2       │ │  Tab 3           │   │ │
│  │  │  Analyzer    │ │  Batch       │ │  About Model     │   │ │
│  │  │  (textarea)  │ │  Dashboard   │ │  (info + limits) │   │ │
│  │  └──────┬───────┘ └──────┬───────┘ └──────────────────┘   │ │
│  │         │ fetch()        │ fetch()                         │ │
│  └─────────┼────────────────┼─────────────────────────────────┘ │
└────────────┼────────────────┼─────────────────────────────────── ┘
             │ HTTP POST      │ HTTP POST (multipart CSV)
             ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI SERVER (Python)                        │
│                   uvicorn backend.main:app                       │
│                                                                  │
│  GET  /api/health  ──► {"status": "ok"}                         │
│  POST /api/analyze ──► run_classifier(text)                      │
│  POST /api/batch   ──► pandas CSV parse → loop → run_classifier  │
│  GET  /*           ──► serve index.html                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ pipeline("text-classification", ...)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│          HUGGING FACE MODEL  (loaded once at startup)            │
│          mohammedlahan/review-sentiment-model                    │
│                                                                  │
│  Architecture : RoBERTa-base                                     │
│  Task         : 3-class text classification                      │
│  Labels       : negative (0)  neutral (1)  positive (2)         │
│  Input limit  : 128 tokens                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decision:** FastAPI serves both the API **and** the HTML file.
This means only one process to run — no separate frontend server needed.

---

## 3. The AI Model

### 3a. What is a Transformer?

A transformer is a type of neural network that reads text by looking at **all words
at the same time** and figuring out how they relate to each other. This is called
"self-attention."

```
Sentence: "The product is not bad at all"

Old models read left-to-right one word at a time:
  The → product → is → not → bad ...  (might think "bad" = negative)

Transformer reads all words simultaneously and links them:
  "not" ←────────────────────────→ "bad"
         (attention score: very high)
  Result: understands this is POSITIVE because "not bad" = good
```

### 3b. What is RoBERTa?

**RoBERTa** (Robustly Optimized BERT Pre-training Approach) is a model released by
Facebook AI in 2019. It was trained on **160 GB of English text** (books, Wikipedia,
news articles, web pages) and learned to understand the English language deeply.

```
RoBERTa-base specs:
  ├── 12 transformer layers
  ├── 12 attention heads per layer
  ├── 768 hidden dimensions
  ├── 50,265 vocabulary tokens
  └── ~125 million parameters (learnable numbers)
```

### 3c. How the Model Was Created (Fine-tuning)

The model started as `roberta-base` (general English understanding) and was
**fine-tuned** on a dataset of labelled customer reviews.

```
PHASE 1 — Pre-training (done by Facebook AI, not us)
─────────────────────────────────────────────────────
  Input:  Huge English text corpus (160 GB)
  Task:   Predict masked words: "The [MASK] is broken" → "product"
  Result: Model learns grammar, facts, word relationships
  Time:   Days on hundreds of GPUs

PHASE 2 — Fine-tuning (done on our review dataset)
────────────────────────────────────────────────────
  Input:  Labelled reviews ("Terrible product" → negative)
  Task:   Predict the correct sentiment label
  Result: Model specialises in sentiment classification
  Time:   Hours on a single GPU

  Training loop:
  ┌─────────────────────────────────────────────────┐
  │  For each batch of reviews:                     │
  │  1. Feed review text into RoBERTa               │
  │  2. Model outputs 3 scores (neg/neu/pos)        │
  │  3. Compare with the true label                 │
  │  4. Calculate error (loss)                      │
  │  5. Adjust the 125M numbers slightly to         │
  │     reduce the error (backpropagation)          │
  │  6. Repeat thousands of times                   │
  └─────────────────────────────────────────────────┘
```

### 3d. The 5 Model Files Explained

```
backend/model/
├── model.safetensors      ← The 125M trained numbers (weights)
│                            ~480 MB, the actual "brain" of the model
│
├── config.json            ← Blueprint of the architecture
│                            Tells: 12 layers, 768 dims, labels={0:negative,...}
│
├── tokenizer.json         ← The vocabulary (50,265 word-pieces)
│                            Maps: "amazing" → [1337, 42] (token IDs)
│
├── tokenizer_config.json  ← Tokenizer settings
│                            Special tokens: <s> (start), </s> (end), <pad>
│
└── training_args.bin      ← Hyper-parameters used during training
                             (learning rate, batch size, epochs — kept for reference)
```

### 3e. How the Model Runs at Inference Time

When you click "Analyze" on a review, this happens inside the model:

```
Step 1 — TOKENIZE
  Input:  "This product is amazing!"
  Output: [0, 713, 1744, 16, 2721, 328, 2]
           ^                              ^
        <s> (start token)         </s> (end token)
  (each number = one word-piece in the vocabulary)

Step 2 — EMBED
  Each token ID → 768-dimensional vector (a list of 768 numbers)
  "amazing" → [0.23, -0.11, 0.87, ... 768 numbers total]
  These vectors encode the initial meaning of each token.

Step 3 — 12 TRANSFORMER LAYERS
  Each layer refines the vectors using self-attention:
  - Every token looks at every other token
  - Builds richer understanding of meaning in context
  After 12 layers, each token vector carries full contextual meaning

Step 4 — CLASSIFICATION HEAD
  Takes the [CLS] (first) token vector → passes through a linear layer
  Output: 3 raw scores (logits): [-1.8,  -0.4,  3.2]
                                  neg    neu    pos

Step 5 — SOFTMAX
  Converts logits to probabilities that sum to 1:
  negative: e^(-1.8) / total = 0.02  (2%)
  neutral:  e^(-0.4) / total = 0.09  (9%)
  positive: e^(3.2)  / total = 0.89  (89%)

Step 6 — OUTPUT
  Winner = positive (highest probability)
  Return: { sentiment: "positive", confidence: 0.89,
            scores: { negative: 0.02, neutral: 0.09, positive: 0.89 } }
```

### 3f. Why truncation=True, max_length=128?

RoBERTa can process a maximum of 512 tokens. Very long reviews would exceed this
and crash the model. We set `max_length=128` (about 90–100 words) for two reasons:

1. **Safety** — prevents any crash
2. **Speed** — processing 128 tokens is 4× faster than 512; most reviews fit in 128

```python
# In backend/main.py:
classifier(text, truncation=True, max_length=128)
#                 ^                 ^
#    chop if too long      keep only first 128 tokens
```

### 3g. Known Limitations

| Limitation | Why it happens | Example |
|---|---|---|
| Neutral is hardest | Mixed signals confuse the model | "Good product but bad shipping" |
| Sarcasm is missed | Model trained mostly on literal text | "Oh great, broken again" |
| Short reviews | Too little context | "Ok" or "👍" |
| Long reviews truncated | 128-token limit | First 100 words only |
| Domain shift | Training data may differ from your domain | B2B software reviews |

---

## 4. The Backend — FastAPI Server

### 4a. File: `backend/main.py`

The entire server is in one Python file. Here's what each section does:

```
backend/main.py
│
├── IMPORTS
│   ├── FastAPI, HTTPException  → web framework
│   ├── CORSMiddleware         → allow browser requests from any origin
│   ├── FileResponse           → send the HTML file to the browser
│   ├── BaseModel (Pydantic)   → automatic input validation
│   ├── pandas                 → parse the uploaded CSV
│   └── pipeline (transformers)→ load and run the AI model
│
├── MODEL LOADING  (runs once at startup)
│   └── pipeline("text-classification", model="mohammedlahan/...")
│
├── APP CREATION
│   ├── app = FastAPI(...)
│   └── app.add_middleware(CORSMiddleware, ...)
│
├── HELPER FUNCTION: run_classifier(text)
│   └── calls the model, formats the result into a clean dict
│
├── ENDPOINT: GET /api/health
│   └── quick ping to check the server is alive
│
├── ENDPOINT: POST /api/analyze
│   ├── validates: text must not be empty
│   └── returns: {sentiment, confidence, scores}
│
├── ENDPOINT: POST /api/batch
│   ├── reads the uploaded CSV file bytes
│   ├── pandas parses it into a DataFrame
│   ├── validates: must have "review" column, max 500 rows
│   ├── loops over every row → runs run_classifier()
│   ├── builds summary: counts, % negative, top 5 negatives
│   └── returns: {results[], summary{}}
│
└── CATCH-ALL ROUTE: GET /*
    └── serves frontend/index.html for any URL not starting with /api/
```

### 4b. Input Validation

FastAPI uses **Pydantic** to validate inputs automatically:

```python
class AnalyzeRequest(BaseModel):
    text: str   # must be a string — FastAPI rejects anything else with 422

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
```

If a user sends `{"text": 123}` → FastAPI returns a clear error automatically.
If a user sends `{"text": ""}` → our code returns "Text cannot be empty."

### 4c. CORS Explained

```
Without CORS:                    With CORS enabled:
─────────────                    ──────────────────
Browser loads page               Browser loads page
 from localhost:8000              from localhost:8000
Tries to call API                Tries to call API
 at localhost:8000                at localhost:8000
Browser BLOCKS it                Browser allows it ✓
 (same-origin policy)             (server said it's OK)
```

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
# allow_origins=["*"] means: accept requests from ANY domain
# In production, change to: allow_origins=["https://yoursite.com"]
```

---

## 5. The Frontend — Building the UI

### 5a. Philosophy: One File, Zero Build Tools

The entire UI lives in `frontend/index.html`. It contains:
- `<style>` block with all CSS (no separate stylesheet)
- HTML structure for all 3 tabs
- `<script>` block with all JavaScript (no separate JS file)

**Why one file?**
- No webpack, no npm, no build step
- FastAPI can serve it with one `FileResponse` line
- Beginners can read and edit it instantly

### 5b. The Tab System

```html
<!-- Three sections — only one visible at a time -->
<section id="tab-analyzer" class="visible">...</section>
<section id="tab-batch">...</section>
<section id="tab-about">...</section>
```

```css
/* CSS controls visibility */
section         { display: none; }    /* hidden by default */
section.visible { display: block; }   /* shown when "visible" class added */
```

```javascript
// JavaScript switches tabs — no page reload
function showTab(name, btn) {
    // hide all
    document.querySelectorAll('section')
            .forEach(s => s.classList.remove('visible'));
    // show chosen one
    document.getElementById('tab-' + name).classList.add('visible');
}
```

This is the **Single-Page Application (SPA)** pattern — the page never reloads,
only sections are shown/hidden.

### 5c. The Colour System

Three colours map to three sentiments — applied via CSS classes:

```
Positive → Green   (#48bb78 bar, #c6f6d5 badge background)
Neutral  → Amber   (#f6ad55 bar, #fefcbf badge background)
Negative → Red     (#fc8181 bar, #fed7d7 badge background)
```

JavaScript adds the right class dynamically:
```javascript
badge.className = 'badge badge-' + data.sentiment;
//  data.sentiment = "positive" → class = "badge badge-positive" → green
//  data.sentiment = "negative" → class = "badge badge-negative" → red
```

### 5d. The Confidence Bars

Each bar is built from two `<div>` elements:

```html
<div class="bar-track">          <!-- grey background track -->
  <div class="bar-fill bar-positive"
       style="width: 89%">       <!-- coloured fill, width = score % -->
  </div>
</div>
```

The CSS transition makes the bar animate smoothly:
```css
.bar-fill { transition: width 0.5s ease; }
```

### 5e. The API Call (fetch)

```javascript
async function analyzeReview() {
    // 1. Show spinner, hide old result
    loading.style.display = 'block';
    result.style.display  = 'none';

    try {
        // 2. POST to our FastAPI backend
        const resp = await fetch('/api/analyze', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text: "review text here" })
        });

        // 3. Parse JSON response
        const data = await resp.json();

        // 4. If server returned an error (4xx/5xx), throw it
        if (!resp.ok) throw new Error(data.detail || 'Server error');

        // 5. Update the DOM with results
        badge.textContent = data.sentiment;      // "positive"
        badge.className   = 'badge badge-' + data.sentiment;

    } catch (err) {
        // 6. Show error to user
        errorDiv.textContent = '⚠️ ' + err.message;
    } finally {
        // 7. Always hide spinner (whether success or error)
        loading.style.display = 'none';
    }
}
```

**`async/await`** makes it possible to wait for the server response without
freezing the browser. Without it, the browser would lock up for the 200ms the
model takes to run.

### 5f. CSV Batch Upload

```javascript
// 1. User picks a file
function handleFileSelect(input) {
    selectedFile = input.files[0];   // browser File object
}

// 2. Upload it
async function uploadBatch() {
    const form = new FormData();
    form.append('file', selectedFile);   // multipart/form-data encoding

    const resp = await fetch('/api/batch', {
        method: 'POST',
        body:   form    // no Content-Type header — browser sets it automatically
    });
}
```

**`FormData`** is the browser's built-in way to send files over HTTP — the same
format used by standard HTML `<form>` file uploads.

### 5g. Chart.js Bar Chart

```javascript
// Loaded from CDN — no installation needed
// <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/..."></script>

new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
        labels:   ['Positive', 'Neutral', 'Negative'],
        datasets: [{
            data:            [42, 15, 8],           // counts from API response
            backgroundColor: ['#48bb78', '#f6ad55', '#fc8181'],  // green/amber/red
        }]
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } }
});
```

### 5h. XSS Protection

User review text is inserted into the HTML table. Without protection, a review
containing `<script>alert('hacked')</script>` could run malicious code.

```javascript
function escapeHtml(str) {
    return String(str)
        .replace(/&/g,  '&amp;')    // & → &amp;
        .replace(/</g,  '&lt;')     // < → &lt;   (prevents HTML tags)
        .replace(/>/g,  '&gt;')     // > → &gt;
        .replace(/"/g,  '&quot;');  // " → &quot;
}

// Used when inserting user text into the table:
td.innerHTML = escapeHtml(review.text);
```

---

## 6. How Everything Connects — Request Lifecycle

### Single review analysis (end to end)

```
TIME    EVENT
─────   ──────────────────────────────────────────────────────────────
0 ms    User types "Great product!" and clicks Analyze

1 ms    JavaScript: spinner shown, result hidden

2 ms    Browser sends HTTP POST to localhost:8000/api/analyze
        Body: {"text": "Great product!"}

3 ms    FastAPI receives the request
        Pydantic validates: text is a non-empty string ✓

4 ms    run_classifier("Great product!") is called

5 ms    Tokenizer converts text to token IDs: [0, 6, 1744, 328, 2]

6 ms    Token IDs pass through 12 RoBERTa transformer layers

150 ms  Classification head outputs logits: [-2.1, -0.8, 3.4]

151 ms  Softmax: {negative: 0.01, neutral: 0.04, positive: 0.95}

152 ms  FastAPI builds response dict:
        {sentiment:"positive", confidence:0.95, scores:{...}}

153 ms  HTTP response (JSON) sent back to browser

154 ms  JavaScript receives the response, parses JSON

155 ms  DOM updated:
        - Badge text set to "positive", class "badge-positive" (green)
        - Confidence "95.0%" shown
        - 3 bars drawn at 1%, 4%, 95% widths

156 ms  Spinner hidden, result section shown

User sees the result instantly ✓
```

---

## 7. Technology Choices Explained

### Why FastAPI and not Flask?

```
Flask (older):                    FastAPI (modern):
──────────────                    ─────────────────
No type hints                     Full type hint support
No automatic validation           Pydantic auto-validates inputs
Synchronous only                  Async/await support (ASGI)
No auto-generated docs            Auto docs at /docs and /redoc
Need to write validators          Request body validated automatically
```

### Why Uvicorn and not running Python directly?

Python alone cannot listen for HTTP connections. You need a server.
Uvicorn is the standard ASGI server for FastAPI — they were designed together.

```
Without Uvicorn:          With Uvicorn:
─────────────────         ─────────────
Can't receive HTTP        Listens on port 8000
Can't handle concurrent   Handles multiple requests at once
  requests                 using async I/O
Can't send HTTP headers   Handles all HTTP protocol details
                          Supports WebSockets too
```

### Why Pandas for CSV parsing?

```python
# Without pandas — fragile:
for line in file.read().decode().split('\n'):
    cols = line.split(',')   # BREAKS if review contains a comma!

# With pandas — robust:
df = pd.read_csv(io.BytesIO(contents))
# Handles: quoted commas, encoding, blank rows, NaN values, headers
```

### Why HuggingFace Hub for the model?

| Option | File size in Git | User experience |
|---|---|---|
| In Git repo directly | +480 MB per commit | Slow clone, hits GitHub 100 MB limit |
| Google Drive link | 0 MB | User must download manually, copy to right folder |
| **HuggingFace Hub** ✅ | 0 MB | Auto-downloads on first run, cached forever |

### Why Chart.js from CDN?

Drawing a bar chart in raw HTML canvas requires ~150 lines of canvas math.
Chart.js does it in 15 lines. CDN = no npm, no build step, cached by browser.

---

## 8. File-by-File Breakdown

```
review-analyzer/
│
├── backend/
│   ├── __init__.py
│   │   └── Empty file — tells Python "backend/ is a package"
│   │       Without this, `uvicorn backend.main:app` would fail
│   │
│   ├── main.py                          (~170 lines)
│   │   ├── Imports
│   │   ├── Path configuration
│   │   ├── Model loading (1 pipeline call)
│   │   ├── FastAPI app creation
│   │   ├── CORS middleware
│   │   ├── run_classifier() helper
│   │   ├── GET  /api/health
│   │   ├── POST /api/analyze
│   │   ├── POST /api/batch
│   │   └── GET  /* (serves index.html)
│   │
│   └── model/
│       └── .gitkeep                     (empty placeholder)
│           The actual model files are on HuggingFace Hub.
│           Downloaded automatically on first run.
│
├── frontend/
│   └── index.html                       (~300 lines)
│       ├── <head>  — Chart.js CDN link, all CSS in <style>
│       ├── <header> — app title
│       ├── <nav>   — 3 tab buttons
│       ├── Tab 1: Analyzer
│       │   ├── Textarea + Analyze button
│       │   ├── 3 example-review buttons
│       │   ├── Loading spinner (shown during API call)
│       │   ├── Error message div
│       │   └── Result: badge + confidence text + 3 bars
│       ├── Tab 2: Batch Dashboard
│       │   ├── File drop zone + Upload button
│       │   ├── Loading spinner
│       │   ├── Summary grid (5 stat cards)
│       │   ├── Chart.js canvas
│       │   ├── Top-5 negative reviews table
│       │   └── Full results table
│       ├── Tab 3: About the Model
│       │   ├── Architecture info
│       │   ├── Accuracy placeholder
│       │   └── Limitations list
│       └── <script> — all JavaScript
│           ├── showTab()
│           ├── fillExample()
│           ├── analyzeReview()
│           ├── handleFileSelect()
│           ├── uploadBatch()
│           ├── pct() helper
│           └── escapeHtml() helper
│
├── sample_reviews.csv                   (20 rows)
│   └── 20 hand-labelled reviews (pos/neu/neg) for testing batch upload
│
├── requirements.txt
│   ├── fastapi==0.115.5
│   ├── uvicorn[standard]==0.32.1
│   ├── pandas==2.2.3
│   ├── transformers==4.47.0
│   ├── torch==2.5.1
│   ├── safetensors==0.4.5
│   ├── tokenizers==0.21.0
│   ├── huggingface_hub==0.26.3
│   └── python-multipart==0.0.12
│
├── .gitignore
│   ├── venv/              ← never commit (100s of MB)
│   ├── __pycache__/       ← Python bytecode, auto-generated
│   ├── *.safetensors      ← model weights (on HuggingFace instead)
│   ├── *.bin              ← training args (on HuggingFace instead)
│   └── .env               ← secrets (never commit)
│
└── README.md
    └── Setup guide, API reference, HuggingFace model link
```

---

*Document version 1.0 — Customer Review Analyzer*
*Model hosted at: https://huggingface.co/mohammedlahan/review-sentiment-model*
*Code hosted at:  https://github.com/mohammedlahan/review-analyzer*
