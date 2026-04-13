# PowerTrust — Malaysia Solar Intelligence System

IE 7374 Generative AI Hackathon | Northeastern University | Spring 2026

## What This Is
A RAG-based AI system that answers questions about distributed solar development in Malaysia across 6 dimensions: Cost, Grid Access, Policy, Utility Standards, Approvals, and Unknown Unknowns.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key
### 3. Build the database
Run `notebooks/01_build_database.ipynb` to build the Chroma vector database.

### 4. Launch the UI
```bash
streamlit run app.py
```

## Stack
- Embedding: BAAI/bge-small-en (HuggingFace)
- Vector DB: Chroma
- LLM: Groq llama-3.3-70b-versatile
- UI: Streamlit
- Web Search: DuckDuckGo (ddgs)

## Country
Malaysia — covering Peninsular Malaysia, Sabah, and Sarawak
