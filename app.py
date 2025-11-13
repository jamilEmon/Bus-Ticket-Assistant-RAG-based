import streamlit as st
import os, json, sqlite3, time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests

st.set_page_config(page_title="Bus Ticket Assistant (RAG)", layout="wide")
st.title("Bus Ticket Assistant — RAG-based (offline)")

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
FAISS_DIR = os.path.join(ROOT, "faiss")
DB_PATH = os.path.join(DATA_DIR, "bookings.db")
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"

# ensure directories
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load or initialize SQLite bookings DB
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            provider TEXT,
            origin TEXT,
            destination TEXT,
            travel_date TEXT,
            created_at REAL
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# Load data.json
def load_data():
    p = os.path.join(DATA_DIR, "data.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"districts": [], "providers": []}

DATA = load_data()

# Embeddings & FAISS utilities
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMB_MODEL_NAME)

def build_faiss_index(texts, ids):
    model = get_embedder()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs).astype("float32"))
    # save meta and index
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
    with open(os.path.join(FAISS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "texts": texts}, f, ensure_ascii=False, indent=2)
    return index

def load_faiss_index():
    idx_path = os.path.join(FAISS_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_DIR, "meta.json")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    return None, None

def index_exists():
    return os.path.exists(os.path.join(FAISS_DIR, "index.faiss")) and os.path.exists(os.path.join(FAISS_DIR, "meta.json"))

# Ingest data: create texts from routes and provider files
def create_corpus_from_data():
    texts = []
    ids = []
    # routes
    for prov in DATA.get("providers", []):
        for r in prov.get("routes", []):
            txt = "Provider: " + prov.get('name') + "\\n" + \
                  "Route: " + r.get('origin') + " -> " + r.get('destination') + "\\n" + \
                  "Fare: " + str(r.get('fare')) + "\\n" + \
                  "Departure: " + r.get('departure')
            texts.append(txt)
            ids.append("route::" + prov.get('name') + "::" + r.get('origin') + "->" + r.get('destination'))
    # provider texts
    pt_dir = os.path.join(DATA_DIR, "provider_texts")
    if os.path.exists(pt_dir):
        for fname in sorted(os.listdir(pt_dir)):
            path = os.path.join(pt_dir, fname)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
                texts.append(txt)
                ids.append("provider::" + fname)
    return texts, ids

# Build index if not exists
if not index_exists():
    with st.spinner("Building vector index (this may download models on first run)..."):
        texts, ids = create_corpus_from_data()
        if texts:
            build_faiss_index(texts, ids)
        else:
            st.info("No data found to index. Place data.json and provider_texts in the data/ folder.")

index, meta = load_faiss_index()

# Generator model (flan-t5-small)
@st.cache_resource(show_spinner=False)
def get_generator():
    gen = pipeline("text2text-generation", model=GEN_MODEL_NAME, max_length=256)
    return gen

gen = get_generator()

# Utility: semantic search
def semantic_search(query, top_k=3):
    if index is None or meta is None:
        return []
    embed = get_embedder().encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(embed).astype("float32"), top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        doc_text = meta["texts"][idx]
        doc_id = meta["ids"][idx]
        results.append({"id": doc_id, "text": doc_text, "score": float(dist)})
    return results

# UI layout: sidebar for data and ingestion
st.sidebar.header("Data & Index")
st.sidebar.markdown("Data is loaded from `data/data.json` and `data/provider_texts/`")
if st.sidebar.button("Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        texts, ids = create_corpus_from_data()
        build_faiss_index(texts, ids)
        index, meta = load_faiss_index()
    st.sidebar.success("Index rebuilt")

st.sidebar.header("Bookings")

# Initialize session state for showing bookings if not already present
if 'show_bookings' not in st.session_state:
    st.session_state.show_bookings = False

if st.sidebar.button("View Bookings"):
    st.session_state.show_bookings = not st.session_state.show_bookings # Toggle visibility

if st.session_state.show_bookings:
    c = conn.cursor()
    rows = c.execute("SELECT id, name, phone, provider, origin, destination, travel_date, created_at FROM bookings ORDER BY id DESC").fetchall()
    if rows:
        for r in rows:
            st.sidebar.markdown(f"**ID {r[0]}** — {r[1]} ({r[2]}) — {r[3]} — {r[4]}→{r[5]} — {r[6]}")
            # Use a unique key for each cancel button, incorporating the booking ID
            if st.sidebar.button(f"Cancel {r[0]}", key=f"cancel_{r[0]}"):
                c.execute("DELETE FROM bookings WHERE id=?", (r[0],))
                conn.commit()
                st.sidebar.success(f"Cancelled booking {r[0]}")
                # Force a rerun to refresh the bookings list
                # st.rerun() # Removed to avoid potential issues within the loop
    else:
        st.sidebar.write("No bookings yet.")

# Main: tabs for Search, Book, Provider
tab1 = st.tabs(["Search Buses", "Book Ticket", "Provider Info", "Sample Q&A"]) [0]

with tab1:
    st.header("Search for Buses")
    query = st.text_input("Enter your search query (e.g., Dhaka to Rajshahi)")
    if query:
        results = semantic_search(query, top_k=5)
        if results:
            st.subheader("Search Results")
            for i, r in enumerate(results):
                st.markdown(f"**Result {i+1}:** {r['id']} (Score: {r['score']})\\n\\n{r['text']}")
        else:
            st.write("No results found.")

tab2 = st.tabs(["Search Buses", "Book Ticket", "Provider Info", "Sample Q&A"]) [1]

with tab2:

    st.header("Book a ticket")
    with st.form("book_form"):
        name = st.text_input("Your name")
        phone = st.text_input("Phone number")
        provider = st.selectbox("Provider", [p['name'] for p in DATA.get('providers',[])])

        origin = st.text_input("Origin (city)", value="Dhaka")
        destination = st.text_input("Destination (city)", value="Rajshahi")
        travel_date = st.date_input("Travel date")
        submitted = st.form_submit_button("Book now")
        if submitted:
            c = conn.cursor()
            c.execute("INSERT INTO bookings (name, phone, provider, origin, destination, travel_date, created_at) VALUES (?,?,?,?,?,?,?)", (
                name, phone, provider, origin, destination, travel_date.isoformat(), time.time()
            ))
            conn.commit()
            st.success("Booking created successfully.")

# Provider Info tab
tab3 = st.tabs(["Search Buses", "Book Ticket", "Provider Info", "Sample Q&A"]) [2]

with tab3:
    st.header("Provider details")
    pname = st.selectbox("Select provider", [p['name'] for p in DATA.get('providers',[])])
    if st.button("Show details for selected provider"):
        # try semantic search first
        res = semantic_search(f"provider {pname}", top_k=3)
        if res:
            for r in res:
                st.markdown(f"**{r['id']}**\\n\\n{r['text']}")
        else:
            # fallback: read file
            path = os.path.join(DATA_DIR, "provider_texts", f"{pname}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    st.text(f.read())
            else:
                st.write("No details found.")

# Sample Q&A tab
tab4 = st.tabs(["Search Buses", "Book Ticket", "Provider Info", "Sample Q&A"]) [3]

with tab4:

    st.header("Ask a free-text question (RAG)")
    q = st.text_input("Your question", value="Are there any buses from Dhaka to Rajshahi under 500 taka?")
    if st.button("Ask"):
        retrieved = semantic_search(q, top_k=4)
        if not retrieved:
            st.warning("No data indexed to answer this question.")
        else:
            prompt = "Use the passages to answer the question: {}\\n Passages:\\n".format(q)
            for r in retrieved:
                prompt += r['text'] + "\\n---\\n"
            prompt += "Answer concisely."
            out = gen(prompt, max_length=200)
            st.markdown("**Answer:** " + out[0]['generated_text'])
            st.subheader("Sources")
            for i, r in enumerate(retrieved):
                st.markdown(f"**Source {i+1}:** {r['id']}**\\n\\n{r['text']}")
