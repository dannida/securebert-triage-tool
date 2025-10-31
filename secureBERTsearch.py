"""
securebert_search.py
--------------------
Use SecureBERT to find past incidents similar to a new alert.
- Builds embeddings for a tiny knowledge base (you'll replace with your own later)
- Searches top-k most similar items via cosine similarity
"""

# ============ 0) Imports ============
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============ 1) Load SecureBERT ============
# - Downloads the tokenizer/model the first time and caches them for reuse
print("[INFO] Loading SecureBERT (first run may download ~500MB)...")
tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = AutoModel.from_pretrained("ehsanaghaei/SecureBERT")
model.eval()  # inference mode (faster, uses less memory)

# ============ 2) Example knowledge base ============
# In real life, replace these strings with your tickets/CVEs/IR notes
kb_items = [
    "CVE-2021-XYZ: Deserialization in YAML leads to remote code execution on admin endpoint.",
    "Web app leaks stack traces; verbose errors reveal framework versions (info disclosure).",
    "TLS disabled on internal API; susceptible to man-in-the-middle attacks.",
    "SQL injection detected on /login via ' OR 1=1 -- payload; WAF blocked attempts.",
    "Prototype pollution in JSON parser allows property injection and possible RCE chain.",
    "Directory traversal lets attackers read /etc/passwd via crafted file paths.",
]

# ============ 3) Helper: mean-pool embeddings ============
def embed_texts(texts):
    """
    Convert a list of strings into L2-normalized sentence embeddings (numpy array).
    - Tokenizes with padding/truncation
    - Runs the model
    - Mean-pools token embeddings with attention mask
    - L2-normalizes so cosine similarity works well
    """
    # Tokenize as a batch
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():  # no gradients = faster + smaller memory
        out = model(**batch)
        last_hidden = out.last_hidden_state  # [B, T, 768]

        # Mask out padding tokens when averaging
        mask = batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)  # [B, 768]

        # Normalize vectors to unit length
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

# ============ 4) Build KB embeddings (once) ============
print("[INFO] Embedding knowledge base items...")
kb_emb = embed_texts(kb_items)  # shape: [N, 768]
print(f"[INFO] Embedded {len(kb_items)} items.")

# ============ 5) Search function ============
def search_similar(query_text, top_k=3):
    """
    Embed the query and compute cosine similarity to all KB items.
    Returns a list of (text, similarity) sorted high->low.
    """
    q_emb = embed_texts([query_text])  # [1, 768]
    sims = cosine_similarity(q_emb, kb_emb)[0]  # [N]
    top_idx = np.argsort(-sims)[:top_k]
    return [(kb_items[i], float(sims[i])) for i in top_idx]

# ============ 6) Demo query ============
query = "Unauthenticated remote code execution via YAML deserialization in the admin API."
print("\nQUERY:")
print(" ", query)

results = search_similar(query, top_k=3)

print("\nTOP MATCHES:")
for rank, (text, score) in enumerate(results, start=1):
    print(f"{rank}. (cosine={score:.3f}) {text}")

# Simple triage guidance from similarity score (tune thresholds on your data)
top_score = results[0][1]
print("\nTRIAGE HINT:")
if top_score > 0.90:
    print("- Looks like a DUPLICATE of a past issue (very similar).")
elif top_score > 0.75:
    print("- Looks RELATED; reuse previous mitigation steps/playbook.")
else:
    print("- Likely NEW/DIFFERENT; investigate from scratch.")
