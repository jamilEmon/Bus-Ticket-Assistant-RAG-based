Demo video : https://drive.google.com/file/d/1sxJhhBbn6B_xzk3toaec1bktcgreH2at/view?usp=drive_link

# Bus Ticket Assistant — Offline RAG Streamlit App

This is a minimal **offline RAG** Streamlit app for a Bus Ticket Assistant.
It uses:
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings (free)
- `google/flan-t5-small` for generation (free Hugging Face model)
- `faiss-cpu` for vector search
- `sqlite3` for storing bookings (self-hosted file)

## Project Overview

This project is an offline Retrieval-Augmented Generation (RAG) application designed to assist users with bus ticket inquiries. It allows users to interact with the system through a Streamlit interface to find information about bus services and potentially make bookings.

The application leverages RAG to provide contextually relevant answers by first retrieving relevant information from a knowledge base and then using a language model to generate a coherent response.

### Architecture Diagram

This diagram provides a more detailed view of the application's components and data flow:

```mermaid
graph TD
    A[User] --> B(Streamlit UI - app.py);

    subgraph Core Application Logic
        B --> C{User Query};
        C --> D[Query Embedding Generation];
        D --> E[FAISS Vector Search];
        E --> F[Retrieve Relevant Documents];
        F --> G[Augment Prompt];
        G --> H[Generation Model (flan-t5-small)];
        H --> I[Generated Response];
    end

    subgraph Data Sources
        J[Provider Texts - data/provider_texts/] --> K(Embedding Model - sentence-transformers);
        L[Booking Data - data/bookings.db] --> M(App Logic);
        N[Other Data - data/data.json] --> M;
    end

    K --> E;
    M --> B;
    I --> B;
    B --> O[Final Output to User];

    %% Styling (optional, but can help clarity)
    classDef data fill:#f9f,stroke:#333,stroke-width:2px;
    classDef model fill:#ccf,stroke:#333,stroke-width:2px;
    classDef process fill:#cfc,stroke:#333,stroke-width:2px;

    class A,O data;
    class B,C,E,G,H,I,J,L,N process;
    class D,K model;
```

**Explanation of Components:**
- **User:** Interacts with the application.
- **Streamlit UI (`app.py`):** Provides the web interface for user interaction.
- **User Query:** The input provided by the user.
- **Query Embedding Generation:** Converts the user's query into a numerical vector using `sentence-transformers`.
- **FAISS Vector Search:** Efficiently searches a vector index (created from provider texts) to find documents semantically similar to the query embedding.
- **Retrieve Relevant Documents:** Fetches the actual text content of the documents identified by FAISS.
- **Augment Prompt:** Combines the user's original query with the retrieved document content to create a more informed prompt for the generation model.
- **Generation Model (flan-t5-small):** Takes the augmented prompt and generates a natural language response.
- **Data Sources:**
    - **Provider Texts (`data/provider_texts/`):** Raw text data used to build the embedding index.
    - **Booking Data (`data/bookings.db`):** SQLite database storing booking information.
    - **Other Data (`data/data.json`):** Additional data that might be used by the application logic.
- **Embedding Model (`sentence-transformers`):** Used for creating vector representations of text.
- **App Logic:** Manages data retrieval, interaction with the database, and orchestrates the RAG pipeline.
- **Final Output to User:** The generated response presented through the Streamlit UI.

## Quick Start Guide for New Users

To get started with the Bus Ticket Assistant, follow these steps:

1.  **Set up a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.
    *   Create a virtual environment (e.g., using `venv`):
        ```bash
        python -m venv venv
        ```
    *   Activate the environment:
        *   On Windows: `.\venv\Scripts\activate`
        *   On macOS/Linux: `source venv/bin/activate`

2.  **Install Dependencies:**
    Once your virtual environment is active, install all the necessary Python packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This command will download and install libraries like Streamlit, sentence-transformers, faiss-cpu, etc.

3.  **Prepare Data (Optional):**
    The application comes with sample data. If you have your own custom data, you can place your `data.json` file in the `data/` directory and any custom provider text files in the `data/provider_texts/` directory.

4.  **Run the Streamlit Application:**
    Start the Streamlit server to launch the application:
    ```bash
    streamlit run app.py
    ```
    This command will start the web server, and the application should become accessible in your default web browser.

5.  **Access the Application:**
    The application will typically be available at `http://localhost:8501`. Open this URL in your web browser to use the Bus Ticket Assistant.

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
