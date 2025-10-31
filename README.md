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
**Windows PowerShell**

## How It Works
- Tokenize with SecureBERT ? mean-pool ? L2-normalize  
- Compute cosine similarity to rank KB entries  
- Show top 3; user chooses a solution to view

## Roadmap
- [ ] FAISS index for large KBs  
- [ ] Export matches + solutions as a report  
- [ ] Gradio web UI

<<<<<<< HEAD
## Contributing
See [CONTRIBUTING.md].
=======
<<<<<<< HEAD
# 4) Run
python secureBERTplaybookSearch.py
CSV Format
id,title,details,solution
CVE-2021-XYZ,YAML deserialization RCE,"Unauthenticated RCE via YAML deserialization.","1) Patch lib ... 2) Disable unsafe_load ... "
How It Works
Tokenize with SecureBERT ? mean-pool ? L2-normalize
=======
## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).
>>>>>>> bdb433c (Fix Markdown formatting in README)
>>>>>>> 5f867c1

## Security
See [SECURITY.md](SECURITY.md).

## License
[MIT](LICENSE)
