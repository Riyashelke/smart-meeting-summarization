# pipeline_combined.py
"""
Single-file pipeline that combines:
 - transcription (whisper)
 - summarization (pegasus/bart or fallback)
 - action-item extraction (rule-based + optional BERT)

Usage (from your venv):
  pip install -r requirements.txt
  python pipeline_combined.py

Place your `index.html` in the same folder (or edit STATIC_DIR).
The frontend's BACKEND_URL should be '/api/analyze' if serving index.html from the same server.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try to import transformers components; if not available we'll fall back to a small summarizer pipeline.
try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
        BartForConditionalGeneration,
        BartTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Optional BERT classifier for action extraction
try:
    from transformers import BertForSequenceClassification, BertTokenizer
    import torch.nn.functional as F
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent  # serve index.html placed next to this script

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")  # tiny, base, small, medium, large
SUMMARIZER_PEGASUS = os.environ.get("PEGASUS_MODEL", "google/pegasus-xsum")  # heavy
SUMMARIZER_BART = os.environ.get("BART_MODEL", "facebook/bart-large-cnn")  # heavy
SUMMARIZER_FALLBACK = os.environ.get("FALLBACK_SUMMARIZER", "sshleifer/distilbart-cnn-12-6")  # lightweight fallback

BERT_ACTION_MODEL = os.environ.get("BERT_MODEL", "bert-base-uncased")  # placeholder; fine-tune for real task
BERT_CONF_THRESHOLD = float(os.environ.get("BERT_CONF_THRESHOLD", 0.6))

MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

ACTION_KEYWORDS = [
    "will",
    "should",
    "need to",
    "needs to",
    "has to",
    "have to",
    "to work on",
    "to complete",
    "to finish",
    "to send",
    "to finalize",
    "to review",
    "to update",
    "assigned to",
    "follow up",
    "next meeting",
    "schedule",
    "must",
    "plan to",
]

SUMMARY_LENGTH_CONFIG = {
    "short": {"max_length": 80, "min_length": 30},
    "medium": {"max_length": 140, "min_length": 60},
    "long": {"max_length": 250, "min_length": 120},
}

ALLOWED = {
    "audio": {
        "mimes": {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/m4a", "audio/mp4", "audio/ogg"},
        "exts": {"mp3", "wav", "m4a", "ogg"},
    },
    "video": {
        "mimes": {"video/mp4", "video/webm", "video/quicktime"},
        "exts": {"mp4", "webm", "mov"},
    },
}

# ---------------------------------------------------------------------------
# Download tokenizers / small dependencies
# ---------------------------------------------------------------------------
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Model loading (best-effort; non-fatal fallbacks)
# ---------------------------------------------------------------------------
print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device set to", DEVICE)

# Summarizer: prefer PEGASUS or BART; fallback to a lightweight pipeline
summarizer_pipeline = None
pegasus_tokenizer = pegasus_model = None
bart_tokenizer = bart_model = None
if TRANSFORMERS_AVAILABLE:
    try:
        print(f"Attempting to load summarizer pipeline ({SUMMARIZER_PEGASUS}) on device {DEVICE}")
        pegasus_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_PEGASUS)
        pegasus_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PEGASUS).to(DEVICE)
        summarizer_backend = "pegasus"
        print("Loaded Pegasus summarizer.")
    except Exception as e:
        print("Pegasus load failed:", e)
        try:
            print(f"Attempting to load BART summarizer ({SUMMARIZER_BART})")
            bart_tokenizer = BartTokenizer.from_pretrained(SUMMARIZER_BART)
            bart_model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_BART).to(DEVICE)
            summarizer_backend = "bart"
            print("Loaded BART summarizer.")
        except Exception as e2:
            print("BART load failed:", e2)
            try:
                print(f"Falling back to pipeline summarizer ({SUMMARIZER_FALLBACK})")
                summarizer_pipeline = hf_pipeline("summarization", model=SUMMARIZER_FALLBACK, device=0 if torch.cuda.is_available() else -1)
                summarizer_backend = "pipeline-fallback"
                print("Loaded fallback summarizer pipeline.")
            except Exception as e3:
                summarizer_backend = None
                print("No summarizer available:", e3)
else:
    print("Transformers not available in environment; summarization will use rule-based fallback.")

# Optional BERT classifier (best-effort)
bert_tokenizer = bert_model = None
if BERT_AVAILABLE:
    try:
        print("Loading BERT classifier for action detection (may be slow)...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_ACTION_MODEL)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_ACTION_MODEL, num_labels=2).to(DEVICE)
        bert_model.eval()
        BERT_LOADED = True
        print("BERT loaded (note: not fine-tuned for action extraction).")
    except Exception as e:
        BERT_LOADED = False
        print("BERT load failed:", e)
else:
    BERT_LOADED = False

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Combined Media Summarizer", version="1.0.0")
# For local testing with index served from same server, allow all origins; tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html (and any assets) from STATIC_DIR as root (html=True).
# This lets you open http://localhost:8000/ and serve index.html automatically.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static_root")


class AnalyzeResponse(BaseModel):
    summary: str
    action_items: List[str]
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers: IO / Validation / Transcription / Summarization / Actions
# ---------------------------------------------------------------------------
def validate_file(upload: UploadFile, media_type: str) -> None:
    if media_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="Invalid mediaType")
    ext = Path(upload.filename or "").suffix.lower().lstrip(".")
    mime_ok = (upload.content_type or "").lower() in ALLOWED[media_type]["mimes"]
    ext_ok = ext in ALLOWED[media_type]["exts"]
    if not (mime_ok or ext_ok):
        raise HTTPException(status_code=400, detail="Unsupported file type for selected mediaType")


async def save_upload(upload: UploadFile, media_type: str) -> Tuple[str, int]:
    validate_file(upload, media_type)
    suffix = Path(upload.filename or "").suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="File too large (max 2GB)")
            tmp.write(chunk)
    finally:
        tmp.close()
    return tmp.name, total


def transcribe_file(path: str, language: Optional[str]) -> Tuple[str, Optional[float]]:
    # Runs whisper synchronous transcribe (fast enough for local use)
    result = whisper_model.transcribe(path, language=language if language else None)
    transcript = (result.get("text") or "").strip()
    duration = None
    try:
        segments = result.get("segments") or []
        if segments:
            duration = float(segments[-1].get("end", 0.0))
    except Exception:
        pass
    return transcript, duration


def summarize_text(text: str, length: str = "short") -> str:
    cfg = SUMMARY_LENGTH_CONFIG.get(length, SUMMARY_LENGTH_CONFIG["short"])
    # prefer pegasus/bart exact models (if loaded), else transformers pipeline, else rule-based shorten
    if pegasus_tokenizer and pegasus_model:
        # encode and generate
        inputs = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=cfg["max_length"], min_length=cfg["min_length"], num_beams=4, no_repeat_ngram_size=3)
        raw = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elif bart_tokenizer and bart_model:
        inputs = bart_tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        summary_ids = bart_model.generate(inputs["input_ids"], max_length=cfg["max_length"], min_length=cfg["min_length"], num_beams=4, no_repeat_ngram_size=3)
        raw = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elif summarizer_pipeline:
        out = summarizer_pipeline(text, max_length=cfg["max_length"], min_length=cfg["min_length"], truncation=True)
        raw = out[0]["summary_text"]
    else:
        # Rule-based fallback: naive extractive — return first N sentences
        sentences = nltk.sent_tokenize(text)
        # choose between 1-4 sentences depending on requested length
        count = {"short": 1, "medium": 2, "long": 4}.get(length, 1)
        raw = " ".join(sentences[: max(1, min(len(sentences), count))])
    # post-process: strip and return
    return raw.strip()


def extract_action_items_from_text(text: str, use_bert: bool = True, conf_threshold: float = BERT_CONF_THRESHOLD) -> List[str]:
    # First, sentence-split
    sentences = nltk.sent_tokenize(text)
    candidates: List[str] = []
    for s in sentences:
        low = s.lower()
        if any(kw in low for kw in ACTION_KEYWORDS):
            candidates.append(s.strip())

    # Optionally run BERT classifier to augment/re-rank candidates
    if use_bert and BERT_LOADED and bert_tokenizer and bert_model:
        extra: List[str] = []
        for s in sentences:
            # Skip very short sentences
            if len(s.split()) < 3:
                continue
            # Tokenize and classify
            inputs = bert_tokenizer(s, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                out = bert_model(**inputs)
                probs = F.softmax(out.logits, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0][pred].item())
            if pred == 1 and conf >= conf_threshold:
                extra.append(f"{s.strip()}  (confidence={conf:.2f})")
        # merge deduped candidates + extra
        combined = []
        seen = set()
        for s in candidates + extra:
            if s not in seen:
                seen.add(s)
                combined.append(s)
        return combined[:50]
    else:
        # return deduped candidates (up to 50)
        seen = set()
        out = []
        for s in candidates:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:50]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    mediaType: str = Form(..., pattern="^(audio|video)$"),
    summaryLength: str = Form("short"),
    extractActions: str = Form("true"),
    language: str = Form("en"),
):
    # 1) persist file to disk
    temp_path, total_bytes = await save_upload(file, mediaType)

    try:
        # 2) transcribe (run in threadpool because whisper may be CPU heavy)
        transcript, duration = await asyncio.get_event_loop().run_in_executor(None, transcribe_file, temp_path, language)
    finally:
        # cleanup file
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if not transcript:
        raise HTTPException(status_code=400, detail="Transcription failed or returned empty text.")

    # 3) summarise
    try:
        summary = await asyncio.get_event_loop().run_in_executor(None, summarize_text, transcript, summaryLength)
    except Exception as e:
        # fallback to simple truncation
        print("Summarization failed:", e)
        summary = " ".join(nltk.sent_tokenize(transcript)[:2])

    # 4) action items
    actions = extract_action_items_from_text(transcript, use_bert=True, conf_threshold=BERT_CONF_THRESHOLD) if extractActions.lower() == "true" else []

    return AnalyzeResponse(summary=summary, action_items=actions, transcript=transcript, duration_seconds=duration)


@app.get("/health")
async def health():
    return {"status": "ok", "summarizer_backend": locals().get("summarizer_backend", None), "bert_loaded": BERT_LOADED}


# ---------------------------------------------------------------------------
# CLI run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # When reloading uvicorn creates watcher + worker; that's normal.
    uvicorn.run("pipeline_combined:app", host="0.0.0.0", port=8000, reload=True)