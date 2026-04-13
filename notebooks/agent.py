import os
import json
from groq import Groq
from ddgs import DDGS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = "/Users/bella/powertrust_rag"
CHROMA_DIR   = os.path.join(BASE_DIR, "chroma_db")
GROQ_API_KEY = "your_groq_api_key_here"

TIME_KEYWORDS = [
    "latest", "current", "recent", "2025", "2026",
    "now", "today", "update", "new", "announce"
]

# ── Embedding + Vectorstore ───────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="powertrust_malaysia"
)

# ── RAG Query ─────────────────────────────────────────────────────────────────
def query_rag(query, region=None, k=5):
    try:
        if region:
            filter_dict = {
                "$and": [
                    {"status": {"$eq": "active"}},
                    {"region": {"$in": [region, "federal"]}}
                ]
            }
        else:
            filter_dict = {"status": {"$eq": "active"}}

        results_with_scores = vectorstore.similarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )
    except Exception:
        results_with_scores = vectorstore.similarity_search_with_score(
            query=query, k=k
        )

    results = [doc for doc, score in results_with_scores]
    scores  = [score for doc, score in results_with_scores]

    alerts = []
    try:
        alert_results = vectorstore.similarity_search(query=query, k=50)
        for doc in alert_results:
            if doc.metadata.get("unknown_unknown_flag") == "True":
                alert = doc.metadata.get("alert_message", "")
                if alert and alert not in alerts:
                    alerts.append(alert)
    except Exception:
        pass

    seen = set()
    sources = []
    for doc in results:
        filename = doc.metadata.get("filename", "")
        title    = doc.metadata.get("title", "Unknown")
        org      = doc.metadata.get("organization", "Unknown")
        if filename and filename not in seen:
            seen.add(filename)
            sources.append({"title": title, "org": org, "filename": filename})

    return results, scores, alerts, sources

# ── Web Search ────────────────────────────────────────────────────────────────
def needs_web_search(query, scores, threshold=0.5):
    top_score = min(scores) if scores else 999
    query_lower = query.lower()
    has_time_keyword = any(kw in query_lower for kw in TIME_KEYWORDS)
    triggered_by_score   = top_score > threshold
    triggered_by_keyword = has_time_keyword
    return triggered_by_score or triggered_by_keyword

def search_web(query, max_results=3):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"Malaysia solar {query}", max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", "")[:300]
                })
    except Exception as e:
        print(f"Web search failed: {e}")
    return results

# ── Prompt Builder ────────────────────────────────────────────────────────────
def format_rag_context(results):
    return "\n\n".join([f"[Document {i+1}]\n{doc.page_content[:800]}"
                        for i, doc in enumerate(results)])

def format_web_context(web_results):
    return "\n\n".join([
        f"[Web Result {i+1}]\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['snippet']}"
        for i, r in enumerate(web_results)
    ])

def build_prompt(query, rag_context, web_context=None, region=None):
    region_note = f"The user is asking about {region} Malaysia." if region else ""
    web_section = f"\n\nSUPPLEMENTARY WEB RESULTS:\n{web_context}" if web_context else ""
    return f"""You are an expert analyst for PowerTrust, helping developers and investors understand distributed solar development in Malaysia.

{region_note}

Answer the user's question using ONLY the context provided below.
- Prioritise information from DATABASE CONTEXT over web results.
- If data is missing, explicitly say so — do not guess.
- Be concise and specific. Use numbers and facts where available.

DATABASE CONTEXT:
{rag_context}
{web_section}

USER QUESTION: {query}"""

# ── Main ask() function ───────────────────────────────────────────────────────
def ask(question, region=None):
    results, scores, alerts, rag_sources = query_rag(question, region=region, k=5)

    web_needed  = needs_web_search(question, scores)
    web_results = search_web(question) if web_needed else []

    rag_context = format_rag_context(results)
    web_context = format_web_context(web_results) if web_results else None
    prompt      = build_prompt(question, rag_context, web_context, region)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {
        "answer":      answer,
        "rag_sources": rag_sources,
        "web_sources": [{"title": r["title"], "url": r["url"]} for r in web_results],
        "alerts":      alerts,
        "web_used":    web_needed
    }