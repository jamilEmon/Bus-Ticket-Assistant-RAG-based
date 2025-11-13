Demo video : https://drive.google.com/file/d/1sxJhhBbn6B_xzk3toaec1bktcgreH2at/view?usp=drive_link
# Bus Ticket Assistant — Offline RAG Streamlit App

This is a minimal **offline RAG** Streamlit app for a Bus Ticket Assistant.
It uses:
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings (free)
- `google/flan-t5-small` for generation (free Hugging Face model)
- `faiss-cpu` for vector search
- `sqlite3` for storing bookings (self-hosted file)

## Quick start (local)

1. Create and activate a Python virtual environment (Python 3.10+ recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you have custom data, place `data/data.json` and provider text files in `data/provider_texts/`
4. Run Streamlit:
   ```bash
   streamlit run app.py
   ```
5. The app will appear at http://localhost:8501

## Files
- `app.py` — main Streamlit app
- `data/` — contains sample `data.json` and provider text files
- `data/faiss` — directory where FAISS index and meta will be created after first run
- `data/bookings.db` — SQLite bookings DB (created automatically)
- `requirements.txt` — Python dependencies

## Notes
- First run may download models (sentence-transformers and flan-t5-small) — this requires internet.
- After models are cached, the app runs offline.
- This is a demo scaffold; for production, consider stronger validation, auth, and more robust storage.
