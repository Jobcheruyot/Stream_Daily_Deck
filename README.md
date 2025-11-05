# DailyDeck — Wow UI (Streamlit)

A production‑ready Streamlit app to:
- Upload **huge CSVs (~500MB)** reliably
- Preview fast with **DuckDB** (no full in‑memory load)
- Render **.ipynb** markdown while **hiding lines starting with `*`**
- Show actions **only** if their label **does not** start with `*`
- Offer **downloads** for filtered markdown and data
- Provide a **clean/wow** UI

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push these files to a GitHub repo.
2. Create a new app on Streamlit Community Cloud and point it to `app.py`.
3. Set the App theme in settings if you wish.

## Deploy (Google Cloud Run — containerized)
- Build a container with Python 3.11 slim, install `requirements.txt`, expose port 8080, and run:
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```
- Make sure to set memory to at least 2–4GB for handling large previews.