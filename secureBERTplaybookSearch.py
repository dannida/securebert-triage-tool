# =========================
# SecureBERT Playbook Search + Interactive Solutions Viewer
# -------------------------
# - Reads a CSV playbook file (kb.csv)
# - Builds embeddings for each item using SecureBERT
# - For a user's query, shows top-K similar matches
# - Lets the user choose a match (1/2/3) to view the solution
# - Loops until the user quits
# =========================

# 0) Imports
from transformers import AutoTokenizer, AutoModel
import torch
import csv
import os
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load SecureBERT (downloaded on first run, cached afterwards)
print("[INFO] Loading SecureBERT...")
tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = AutoModel.from_pretrained("ehsanaghaei/SecureBERT")
model.eval()  # inference mode (faster, less memory)

# 2) Helper: read the knowledge base (playbook) from CSV
def load_kb_csv(path: str) -> List[Dict]:
    """
    Loads a CSV file with columns: id, title, details, solution.
    Returns a list of dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Create it in the current folder with columns: id,title,details,solution"
        )

    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "title", "details", "solution"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must have columns: {', '.join(sorted(required))}. Found: {reader.fieldnames}"
            )
        for row in reader:
            # Clean up whitespace; skip empty rows
            if not (row["title"] or row["details"]):
                continue
            items.append({
                "id": row["id"].strip(),
                "title": row["title"].strip(),
                "details": row["details"].strip(),
                "solution": row["solution"].strip()
            })
    if not items:
        raise ValueError("The CSV loaded but contains no usable rows.")
    return items

# 3) Helper: turn texts into SecureBERT embeddings (mean-pooled, L2-normalized)
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Converts a list of strings into L2-normalized sentence embeddings (numpy array).
    - Tokenizes with padding/truncation
    - Runs SecureBERT
    - Mean-pools token embeddings with attention mask
    - L2-normalizes for cosine similarity
    """
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
        last_hidden = out.last_hidden_state  # [B, T, 768]
        mask = batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)  # [B, 768]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

# 4) Load the KB and build embeddings (title + details give best context)
kb_path = "kb.csv"      # change if your file lives elsewhere
kb_items = load_kb_csv(kb_path)
kb_texts = [f"{it['title']} || {it['details']}" for it in kb_items]
print(f"[INFO] Loaded {len(kb_items)} playbook items from {kb_path}.")

print("[INFO] Embedding playbook items (first time may take a moment)...")
kb_emb = embed_texts(kb_texts)  # shape: [N, 768]
print("[INFO] Done embedding.")

# 5) Search function: get top-K similar items
def search_similar(query_text: str, top_k: int = 3):
    """
    Embed the query and compute cosine similarity to all KB items.
    Returns indices sorted by similarity (high->low), and the scores.
    """
    q_emb = embed_texts([query_text])  # [1, 768]
    sims = cosine_similarity(q_emb, kb_emb)[0]  # [N]
    order = np.argsort(-sims)[:top_k]
    return order, sims

# 6) Pretty-print a result row
def show_result(rank: int, item: Dict, score: float):
    print(f"{rank}. [{item['id']}] {item['title']}")
    print(f"    Similarity: {score:.3f}")
    # Keep details compact here; user can open solution separately
    print(f"    Details: {item['details'][:240]}{'...' if len(item['details'])>240 else ''}\n")

# 7) Interactive loop:
#    - user enters a query
#    - we show top 3 matches
#    - user picks 1/2/3 to see SOLUTION
#    - user can keep picking, or type 'n' for new query, or 'q' to quit.

def interactive_search():
    print("\n=== SecureBERT Playbook Search ===")
    print("Type a short alert/incident description (or 'q' to quit).")
    while True:
        query = input("\nQuery> ").strip()
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        # Search KB
        top_idx, sims = search_similar(query, top_k=3)
        print("\nTop matches:")
        for i, idx in enumerate(top_idx, start=1):
            show_result(i, kb_items[idx], float(sims[idx]))

        # Inner loop to open solutions or start a new query
        while True:
            choice = input("Open solution [1/2/3], (n)ew query, or (q)uit: ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                print("Goodbye!")
                return
            if choice in {"n", "new"}:
                # break to outer loop for a new query
                break
            if choice in {"1", "2", "3"}:
                # Show the solution for the chosen match
                i = int(choice) - 1
                if i < 0 or i >= len(top_idx):
                    print("Invalid choice. Try 1, 2, or 3.")
                    continue
                idx = top_idx[i]
                item = kb_items[idx]
                print("\n-------------------------")
                print(f"SOLUTION for [{item['id']}] {item['title']}")
                print("-------------------------")
                print(item["solution"])
                print("-------------------------\n")
            else:
                print("Please enter 1, 2, 3, 'n' for new query, or 'q' to quit.")

# 8) Start the interactive session
interactive_search()
