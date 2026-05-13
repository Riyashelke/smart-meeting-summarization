import glob
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import spacy
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

from pipeline_config import (
    BART_MAX_INPUT_TOKENS,
    BART_MODEL_NAME,
    BART_OUTPUT,
    GRAPH_OUTPUT,
    PEGASUS_MAX_INPUT_TOKENS,
    PEGASUS_MODEL_NAME,
    PEGASUS_OUTPUT,
    TARGET_SENTENCE_COUNT,
    TEXT_INPUT_FOR_SUMMARY,
    TRANSCRIPT_DIR,
    MAX_OUTPUT_TOKENS,
)

# Download NLTK punkt tokenizer
nltk.download("punkt", quiet=True)

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# =====================================================
# PREPROCESSING FUNCTION
# =====================================================
def preprocess_text(text):

    text = re.sub(r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?\b", " ", text)
    text = re.sub(r"\[(.*?)\]|\((.*?)\)", " ", text)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)

    fillers = r"\b(uh|um|erm|hmm|like|you know|I mean|sort of|kind of)\b"
    text = re.sub(fillers, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# NLP Extraction (Attendees, Budget, Actions)
# =====================================================
def extract_nlp_fields(text):
    doc = nlp(text)

    attendees = list({ent.text for ent in doc.ents if ent.label_ == "PERSON"})
    budget_items = list({ent.text for ent in doc.ents if ent.label_ in ["MONEY", "QUANTITY"]})

    action_keywords = r"\b(will|should|need to|to work on|assigned|next meeting|schedule|must|plan to)\b"
    action_items = [
        sent.text.strip()
        for sent in doc.sents
        if re.search(action_keywords, sent.text, re.IGNORECASE)
    ]

    return attendees, budget_items, action_items


# =====================================================
# GENERIC SUMMARIZER
# =====================================================
def generate_summary(text, tokenizer, model, device, max_tokens):

    inputs = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=MAX_OUTPUT_TOKENS,
        min_length=30,
        no_repeat_ngram_size=3
    )

    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    sentences = nltk.sent_tokenize(raw_summary)
    final_summary = " ".join(sentences[:TARGET_SENTENCE_COUNT])

    return final_summary


def run_summarization(
    transcript_folder: Path = TEXT_INPUT_FOR_SUMMARY if TEXT_INPUT_FOR_SUMMARY.exists() else TRANSCRIPT_DIR,
    pegasus_out: Path = PEGASUS_OUTPUT,
    bart_out: Path = BART_OUTPUT,
    graph_out: Path = GRAPH_OUTPUT,
) -> dict[str, list]:
    """
    Summarize all transcripts in transcript_folder with PEGASUS and BART.
    Returns timing data for plotting.
    """
    print("Loading PEGASUS and BART models...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pegasus_tokenizer = AutoTokenizer.from_pretrained(PEGASUS_MODEL_NAME)
    pegasus_model = AutoModelForSeq2SeqLM.from_pretrained(PEGASUS_MODEL_NAME).to(device)

    bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME).to(device)

    pegasus_out.mkdir(parents=True, exist_ok=True)
    bart_out.mkdir(parents=True, exist_ok=True)

    file_paths = glob.glob(str(transcript_folder / "*.txt"))
    if not file_paths:
        print("❌ No files found.")
        return {"pegasus_times": [], "bart_times": [], "labels": []}

    print(f"Found {len(file_paths)} files.\n")
    pegasus_times: list[float] = []
    bart_times: list[float] = []
    file_labels: list[str] = []

    for path in sorted(file_paths):
        filename = os.path.basename(path)
        print(f"Processing: {filename}")

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        text = preprocess_text(raw_text)
        file_labels.append(filename)

        # PEGASUS
        t1 = time.time()
        pegasus_summary = generate_summary(
            text, pegasus_tokenizer, pegasus_model, device, PEGASUS_MAX_INPUT_TOKENS
        )
        t2 = time.time()
        pegasus_times.append(t2 - t1)

        attendees, budget_items, action_items = extract_nlp_fields(text)

        pegasus_structured = (
            f"📝 Meeting Summary: {filename}\n\n"
            f"👥 Attendees and Roles\n"
            f"• {', '.join(attendees) if attendees else 'Not available'}\n\n"
            f"💰 Budget / Financial Mentions\n"
            f"• {', '.join(budget_items) if budget_items else 'Not mentioned'}\n\n"
            f"💡 Key Design / Discussion Points\n"
            f"• {pegasus_summary}\n\n"
            f"➡️ Next Steps\n"
            + ("\n".join([f"• {a}" for a in action_items]) if action_items else "• Not mentioned")
        )

        with open(pegasus_out / f"PEGASUS_{filename}", "w", encoding="utf-8") as f:
            f.write(pegasus_structured)

        # BART
        t3 = time.time()
        bart_summary = generate_summary(text, bart_tokenizer, bart_model, device, BART_MAX_INPUT_TOKENS)
        t4 = time.time()
        bart_times.append(t4 - t3)

        bart_structured = (
            f"📝 Meeting Summary: {filename}\n\n"
            f"👥 Attendees and Roles\n"
            f"• {', '.join(attendees) if attendees else 'Not available'}\n\n"
            f"💰 Budget / Financial Mentions\n"
            f"• {', '.join(budget_items) if budget_items else 'Not mentioned'}\n\n"
            f"💡 Key Design / Discussion Points\n"
            f"• {bart_summary}\n\n"
            f"➡️ Next Steps\n"
            + ("\n".join([f"• {a}" for a in action_items]) if action_items else "• Not mentioned")
        )

        with open(bart_out / f"BART_{filename}", "w", encoding="utf-8") as f:
            f.write(bart_structured)

        print(f"✔ Finished {filename}\n")

    # Graph
    plt.figure(figsize=(10, 6))
    plt.plot(file_labels, pegasus_times, marker="o", label="Pegasus Time (sec)")
    plt.plot(file_labels, bart_times, marker="o", label="BART Time (sec)")
    plt.xlabel("Files")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Model Performance: Pegasus vs BART")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(graph_out)
    print("\n📊 Performance graph saved as:", graph_out)
    print("✨ All summaries generated successfully!\n")

    return {"pegasus_times": pegasus_times, "bart_times": bart_times, "labels": file_labels}


if __name__ == "__main__":
    run_summarization()