# SecureBERT Triage Tool

This project uses the SecureBERT model to identify and match new security alerts or incidents
against a knowledge base (playbook) of past issues and solutions.

## Features
- Search alerts and return the top 3 similar past incidents.
- Display recommended mitigations directly from the playbook.
- CSV-based knowledge base, easily updatable by analysts.

## Usage
```bash
python secureBERTplaybookSearch.py
