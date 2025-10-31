# SecureBERT Triage Tool

SecureBERT-powered incident triage: paste an alert, get the top similar past incidents and their mitigations from your playbook.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

## Features
- ?? Semantic search over your IR / playbook entries (SecureBERT embeddings)
- ?? Top-k matches with similarity scores
- ?? One-key view of suggested mitigations from your playbook
- ?? CSV knowledge base (easy to update)

## Quick Start
```bash
# 1) Create venv (Windows PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add your playbook as kb.csv with columns:
#   id,title,details,solution
#   (see kb_sample.csv)

## How It Works
- Tokenize with SecureBERT ? mean-pool ? L2-normalize  
- Compute cosine similarity to rank KB entries  
- Show top 3; user chooses a solution to view

## Roadmap
- [ ] FAISS index for large KBs  
- [ ] Export matches + solutions as a report  
- [ ] Gradio web UI

# 4) Run
python secureBERTplaybookSearch.py

## Security
See [SECURITY.md](SECURITY.md).

## License
[MIT](LICENSE)
