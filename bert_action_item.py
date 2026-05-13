import os
from pathlib import Path

import nltk
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

from pipeline_config import (
    BERT_ACTION_OUTPUT,
    BERT_CONF_THRESHOLD,
    BERT_MODEL_NAME,
    PEGASUS_OUTPUT,
    SUMMARY_FOR_ACTIONS,
)

# Allow running on CPU / CUDA / MPS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Rule-based keywords (optional filter)
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
]

nltk.download("punkt", quiet=True)

# ------------------------- LOAD BERT MODEL -------------------------
print("Loading Action Item Extraction Model...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
# For demonstration, we use a pre-trained BERT with a single linear head.
# For proper classification, fine-tune on an action item dataset.
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
model.to(DEVICE)
model.eval()

LABELS = ["Not Action Item", "Action Item"]


# ------------------------- FUNCTION: CLASSIFY SENTENCE -------------------------
def classify_sentence(sentence: str) -> tuple[str, float]:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    return LABELS[pred_id], confidence


# ------------------------- FUNCTION: RULE-BASED FILTER (OPTIONAL) -------------------------
def matches_rule(sentence: str) -> bool:
    for kw in ACTION_KEYWORDS:
        if kw in sentence.lower():
            return True
    return False


# ------------------------- MAIN PIPELINE -------------------------
def extract_actions_from_folder(
    summaries_folder: Path = SUMMARY_FOR_ACTIONS if SUMMARY_FOR_ACTIONS.exists() else PEGASUS_OUTPUT,
    output_folder: Path = BERT_ACTION_OUTPUT,
    conf_threshold: float = BERT_CONF_THRESHOLD,
) -> list[Path]:
    """
    Extract action items from text summaries in summaries_folder.
    Saves *_actions.txt files in output_folder and returns their paths.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(summaries_folder) if f.endswith(".txt")]
    print(f"Found {len(files)} summarized files.\n")
    written: list[Path] = []

    for fname in files:
        print(f"Processing: {fname}")
        with open(Path(summaries_folder) / fname, "r", encoding="utf-8") as f:
            text = f.read().strip()

        sentences = nltk.sent_tokenize(text)
        action_items = []

        for sent in sentences:
            label, conf = classify_sentence(sent)
            # Combine BERT + rule-based filtering
            if label == "Action Item" or matches_rule(sent):
                if conf > conf_threshold:
                    action_items.append(f"{sent}  (confidence={conf:.2f})")

        out_path = output_folder / fname.replace(".txt", "_actions.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            if action_items:
                for t in action_items:
                    out.write("• " + t + "\n")
            else:
                out.write("• No actionable tasks found.\n")

        written.append(out_path)
        print(f"→ Saved action items to {out_path}\n")

    print("✅ All files processed!")
    return written


if __name__ == "__main__":
    extract_actions_from_folder()