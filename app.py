import streamlit as st
import sys
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PowerTrust — Malaysia Solar Intelligence",
    page_icon="☀️",
    layout="wide"
)

# ── Google Drive Path ─────────────────────────────────────────────────────────
# CHROMA_PATH = "/content/drive/.shortcut-targets-by-id/1MdK2VZM_imqTJTfopqamlRJvID4o3GVG/Hackathon/RAG/chroma_db"
# AGENT_PATH  = "/content/drive/.shortcut-targets-by-id/1MdK2VZM_imqTJTfopqamlRJvID4o3GVG/Hackathon/RAG"
# ENV_PATH    = "/content/drive/.shortcut-targets-by-id/1MdK2VZM_imqTJTfopqamlRJvID4o3GVG/Hackathon/RAG/.env"

CHROMA_PATH = "/Users/bella/powertrust_rag/chroma_db"
AGENT_PATH  = "/Users/bella/powertrust_rag"
ENV_PATH    = "/Users/bella/powertrust_rag/.env"

# ── Load API Key ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ENV_PATH)

if AGENT_PATH not in sys.path:
    sys.path.insert(0, AGENT_PATH)

# ── Load RAG Components ───────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="powertrust_malaysia"
    )
    return vectorstore


def run_ask(query, region, vectorstore, history=None):
    
    """
    Direct implementation using already-loaded vectorstore.
    Mirrors ask() from 02_agent.ipynb without re-loading models.
    """
    from groq import Groq
    from dotenv import load_dotenv
    from ddgs import DDGS

    load_dotenv(ENV_PATH, override=True)
    groq_key = os.environ.get("GROQ_API_KEY", "")

    TIME_KEYWORDS = ["latest", "current", "recent", "2025", "2026",
                     "now", "today", "update", "new", "announce"]

    # Step 1 — RAG query
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
            query=query, k=5, filter=filter_dict
        )
    except Exception:
        results_with_scores = vectorstore.similarity_search_with_score(
            query=query, k=5
        )

    results = [doc for doc, score in results_with_scores]
    scores  = [score for doc, score in results_with_scores]

    # Step 2 — Proactive alerts
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

    # Step 3 — Sources
    seen = set()
    rag_sources = []
    for doc in results:
        fname = doc.metadata.get("filename", "")
        if fname and fname not in seen:
            seen.add(fname)
            rag_sources.append({
                "title":    doc.metadata.get("title", "Unknown"),
                "org":      doc.metadata.get("organization", ""),
                "filename": fname
            })

    # Step 4 — Web search if needed
    top_score = min(scores) if scores else 999
    query_lower = query.lower()
    web_needed = any(kw in query_lower for kw in TIME_KEYWORDS) or top_score > 0.5

    web_results = []
    if web_needed:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(f"Malaysia solar {query}", max_results=3):
                    web_results.append({
                        "title": r.get("title", ""),
                        "url":   r.get("href", ""),
                        "snippet": r.get("body", "")[:300]
                    })
        except Exception:
            pass
#    f"[Document {i+1}]\n{doc.page_content[:800]}"


    # Step 5 — Build prompt
    rag_context = "\n\n".join([f"[Source: {doc.metadata.get('organization', 'Unknown')}]\n{doc.page_content[:800]}"
                                for i, doc in enumerate(results)])
    web_context = ""
    if web_results:
        web_context = "\n\nSUPPLEMENTARY WEB RESULTS:\n" + "\n\n".join([
            f"[Web {i+1}]\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['snippet']}"
            for i, r in enumerate(web_results)
        ])

    region_note = f"The user is asking about {region} Malaysia." if region else ""
    prompt = f"""You are an expert analyst for PowerTrust, helping developers and investors understand distributed solar development in Malaysia.

{region_note}

Answer the user's question using ONLY the context provided below.
- Prioritise information from DATABASE CONTEXT over web results.
- If data is missing, explicitly say so — do not guess.
- Be concise and specific. Use numbers and facts where available. When cost data includes a range, always state the full range (e.g. $33–61/MWh), not a single number.- When citing information, mention the source organisation name (e.g. "According to BloombergNEF..." or "As noted by SEDA...").
- If the question is unrelated to Malaysia solar development, politely say you can only help with solar development questions in Malaysia.


DATABASE CONTEXT:
{rag_context}
{web_context}

USER QUESTION: {query}"""

    # Step 6 — Call Groq
    # Build history messages (last 3 exchanges only, text content only)
    history_messages = []
    if history:
        for msg in history[-6:]:
            if msg["role"] == "user":
                history_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                history_messages.append({"role": "assistant", "content": msg.get("content", "")})

    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history_messages + [{"role": "user", "content": prompt}],
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F0FFF4; }

    .stat-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
        border: 1px solid #C6F6D5;
        box-shadow: 0 2px 8px rgba(45,139,78,0.08);
        height: 100%;
    }
    .stat-card-green { border-left: 4px solid #2D8B4E; }
    .stat-card-yellow { border-left: 4px solid #D69E2E; }
    .stat-card-red { border-left: 4px solid #E53E3E; }
    .stat-number { font-size: 24px; font-weight: bold; color: #1A1A1A; margin-bottom: 2px; }
    .stat-label { font-size: 13px; font-weight: 600; color: #2D8B4E; margin-bottom: 4px; }
    .stat-desc { font-size: 12px; color: #718096; line-height: 1.4; }

    .section-title {
        font-size: 12px; font-weight: 600; color: #2D8B4E;
        text-transform: uppercase; letter-spacing: 0.8px; margin: 16px 0 6px 0;
    }

    .user-bubble {
        background-color: #2D8B4E; color: #ffffff;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px; margin: 8px 0;
        max-width: 80%; margin-left: auto; font-size: 15px;
    }
    .assistant-bubble {
        background-color: #FFFFFF; color: #1A1A1A;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px; margin: 8px 0;
        max-width: 80%; font-size: 15px;
        border-left: 3px solid #2D8B4E;
        border: 1px solid #C6F6D5;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .alert-box {
        background-color: #FFFBEB;
        border: 1px solid #F6AD55;
        border-left: 4px solid #D69E2E;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
        color: #7B341E; font-size: 14px;
    }
    .alert-title {
        font-weight: bold; color: #C05621; margin-bottom: 6px;
        font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;
    }

    .source-box {
        background-color: #F7FAFC; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 10px 14px; margin: 4px 0;
        font-size: 13px; color: #4A5568;
    }
    .source-db { border-left: 3px solid #2D8B4E; }
    .source-web { border-left: 3px solid #3182CE; }
    .source-label {
        font-size: 11px; font-weight: bold;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .source-db .source-label { color: #2D8B4E; }
    .source-web .source-label { color: #3182CE; }

    .main-title { font-size: 28px; font-weight: bold; color: #1A1A1A; }
    .main-subtitle { font-size: 14px; color: #718096; margin-top: 4px; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stButton > button {
        background-color: #FFFFFF; color: #1A1A1A;
        border: 1px solid #C6F6D5; border-radius: 8px; font-size: 13px;
    }
    .stButton > button:hover {
        background-color: #F0FFF4; border-color: #2D8B4E; color: #2D8B4E;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div class="main-title">☀️ PowerTrust</div>
    <div class="main-subtitle">Malaysia Distributed Solar Intelligence — Ask anything about solar development in Malaysia</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    region = st.selectbox(
        "Region",
        options=["All regions", "peninsular", "sabah", "sarawak", "federal"],
        index=0,
        help="Filter results by Malaysian region"
    )
    if region == "All regions":
        region = None

    st.markdown("---")
    st.markdown("### 📍 Region Guide")
    st.markdown("""
- **Peninsular** — TNB, SEDA, federal programmes
- **Sabah** — SESB, ECoS regulations
- **Sarawak** — SEB, SET-P framework
- **Federal** — National policy, all regions
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
This system queries a curated database of **37 Malaysian solar documents** across 6 dimensions:

1. Cost & Economics
2. Grid Access
3. Policy & Subsidies
4. Utility Standards
5. Approvals & EIA
6. Unknown Unknowns
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Dashboard Data ────────────────────────────────────────────────────────────
DASHBOARD_DATA = {
    None: {  # All regions
        "title": "Malaysia Solar at a Glance",
        "row1": [
            ("green",  "5,777 MW",    "Total Solar Installed",  "Enough to power ~1.4M Malaysian homes"),
            ("green",  "31%",         "RE Share (2025)",        "1 in 3 MW of electricity is now renewable"),
            ("green",  "$33–61/MWh",  "Solar LCOE",             "Solar is now cheaper than coal ($56–77) & gas ($78–89)"),
            ("green",  "70% by 2050", "Malaysia RE Target",     "National Energy Transition Roadmap (NETR) goal"),
        ],
        "row2_title": "What Developers Should Know",
        "row2": [
            ("yellow", "4 months",   "Typical EIA Approval",  "Real case: 50MW solar farm in Gopeng, Perak (2022)"),
            ("red",    "150 MW",     "Sabah LSS Quota",       "Only 7% of Peninsular's 2,060MW — may already be exhausted"),
            ("green",  "5–7 years",  "Typical Payback",       "Based on 500kWp system at RM 2,100/kWp market rate"),
            ("red",    "300 MW",     "Singapore Export Cap",  "Despite a 1GW interconnection line — 70% sits idle"),
        ],
    },
    "peninsular": {
        "title": "Peninsular Malaysia — Solar Overview",
        "row1": [
            ("green",  "2,060 MW",    "LSS Quota",              "Largest quota allocation across all three regions"),
            ("green",  "40 sen/kWh",  "Electricity Tariff",     "Highest across 3 regions — best revenue for solar projects"),
            ("green",  "$33–61/MWh",  "Solar LCOE",             "Solar is now cheaper than coal ($56–77) & gas ($78–89)"),
            ("green",  "5,777 MW",    "Total Solar Installed",  "Most of Malaysia's solar capacity is in Peninsular"),
        ],
        "row2_title": "Peninsular — Key Numbers for Developers",
        "row2": [
            ("yellow", "4 months",   "Typical EIA Approval",  "Real case: 50MW solar farm in Gopeng, Perak (2022)"),
            ("red",    "300 MW",     "Singapore Export Cap",  "Despite a 1GW interconnection line — 70% sits idle"),
            ("green",  "5–7 years",  "Typical Payback",       "Based on 500kWp system at RM 2,100/kWp market rate"),
            ("yellow", "51%",        "MY Equity Required",    "Minimum Malaysian equity for LSS programme participation"),
        ],
    },
    "sabah": {
        "title": "Sabah — Solar Overview",
        "row1": [
            ("yellow", "60 MW",       "Solar Installed",       "Only 60MW installed out of 150MW quota — market still early"),
            ("red",    "150 MW",      "LSS Quota",             "⚠️ Only 7% of Peninsular's 2,060MW — near exhaustion"),
            ("yellow", "34.5 sen/kWh","Electricity Tariff",    "Lower than Peninsular (40 sen) — affects project revenue"),
            ("yellow", "ECoS",        "Grid Regulator",        "Energy Commission of Sabah — separate from federal SEDA"),
        ],
        "row2_title": "Sabah — Key Risks for Developers",
        "row2": [
            ("red",    "203–286 min", "Grid SAIDI",            "4–6x worse than Peninsular (47 min) — high curtailment risk"),
            ("red",    "7%",          "Quota vs Peninsular",   "Sabah's 150MW is only 7% of Peninsular's 2,060MW quota"),
            ("yellow", "SESB Grid",   "Separate Utility",      "Sabah Electricity Sdn Bhd — different connection process"),
            ("yellow", "No ATAP",     "Federal Prog N/A",      "Solar ATAP and NEM do not apply — use ECoS framework"),
        ],
    },
    "sarawak": {
        "title": "Sarawak — Solar Overview",
        "row1": [
            ("red",    "No ATAP",     "Federal Prog N/A",      "⚠️ Solar ATAP & LSS do not apply — engage SEB directly"),
            ("yellow", "28 sen/kWh",  "Electricity Tariff",    "30% lower than Peninsular — directly impacts project revenue"),
            ("green",  "60% by 2030", "Sarawak RE Target",     "SET-P: state's own energy transition policy (2025)"),
            ("green",  "10 GW",       "Capacity Target 2030",  "Sarawak's own target under SET-P framework"),
        ],
        "row2_title": "Sarawak — Key Numbers for Developers",
        "row2": [
            ("red",    "SET-P 2025",  "State Framework",       "Sarawak Energy Transition Policy — separate from federal NETR"),
            ("green",  "1,500 MW",    "Solar Target 2030",     "Floating solar at Bakun Dam is a key opportunity"),
            ("yellow", "SEB",         "Grid Operator",         "Sarawak Energy Berhad — different approval process & fees"),
            ("red",    "3 REC Systems","Fragmented Market",    "Sarawak RECs cannot be used in Peninsular — non-tradeable"),
        ],
    },
    "federal": {
        "title": "Federal Policy — National Overview",
        "row1": [
            ("green",  "70% by 2050", "National RE Target",    "NETR goal — from current 31% to 70% renewable capacity"),
            ("green",  "RM 1,300B",   "Total Investment 2050", "Required investment to achieve net-zero by 2050"),
            ("green",  "310,000",     "Green Jobs by 2050",    "NETR projects 310,000 green jobs across Malaysia"),
            ("green",  "Net Zero",    "Target Year: 2050",     "Malaysia committed to carbon neutrality by 2050"),
        ],
        "row2_title": "Federal — Key Policy Numbers",
        "row2": [
            ("green",  "10 Projects", "NETR Flagship",         "10 catalyst projects across 6 energy transition levers"),
            ("green",  "RM 45B",      "TNB Grid Investment",   "Grid modernisation investment 2025–2027 under RP4"),
            ("green",  "23,000",      "Jobs by 2025",          "Near-term job creation target under NETR Part 1"),
            ("yellow", "51%",         "MY Equity Required",    "Minimum Malaysian equity for LSS programme participation"),
        ],
    },
}

# ── Dashboard ─────────────────────────────────────────────────────────────────
# region is defined in sidebar — use it here
_region_key = region  # may be None for All regions
_data = DASHBOARD_DATA.get(_region_key, DASHBOARD_DATA[None])

st.markdown(f'<div class="section-title">📊 {_data["title"]}</div>', unsafe_allow_html=True)

row1_cols = st.columns(4)
for col, (color, number, label, desc) in zip(row1_cols, _data["row1"]):
    with col:
        st.markdown(f"""
        <div class="stat-card stat-card-{color}">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f'<div class="section-title">⚠️ {_data["row2_title"]}</div>', unsafe_allow_html=True)

row2_cols = st.columns(4)
for col, (color, number, label, desc) in zip(row2_cols, _data["row2"]):
    with col:
        st.markdown(f"""
        <div class="stat-card stat-card-{color}">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Regional Investment Overview (All regions only) ───────────────────────────
if region is None:
    st.markdown('<div class="section-title">🗺️ Regional Investment Overview</div>', unsafe_allow_html=True)

    map_col, chart_col = st.columns([1, 1])

    # ── Folium Map ────────────────────────────────────────────────────────────
    with map_col:
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[4.5, 109.5], zoom_start=5, tiles="CartoDB positron")

            regions_map = [
                {
                    "name": "Peninsular Malaysia",
                    "lat": 3.8, "lon": 109.0,
                    "color": "green",
                    "popup": "<b>Peninsular Malaysia</b><br><table style='font-size:13px'><tr><td>Tariff</td><td><b>40 sen/kWh</b></td></tr><tr><td>LSS Quota</td><td><b>2,060 MW</b></td></tr><tr><td>ATAP</td><td><b>✅ Available</b></td></tr><tr><td>SAIDI</td><td><b>47.88 min</b></td></tr><tr><td>Regulator</td><td><b>SEDA / TNB</b></td></tr></table>"
                },
                {
                    "name": "Sabah",
                    "lat": 5.5, "lon": 117.0,
                    "color": "orange",
                    "popup": "<b>Sabah</b><br><table style='font-size:13px'><tr><td>Tariff</td><td><b>34.5 sen/kWh</b></td></tr><tr><td>LSS Quota</td><td><b>150 MW ⚠️</b></td></tr><tr><td>ATAP</td><td><b>❌ Not Available</b></td></tr><tr><td>SAIDI</td><td><b>203–286 min</b></td></tr><tr><td>Regulator</td><td><b>ECoS / SESB</b></td></tr></table>"
                },
                {
                    "name": "Sarawak",
                    "lat": 2.5, "lon": 113.5,
                    "color": "red",
                    "popup": "<b>Sarawak</b><br><table style='font-size:13px'><tr><td>Tariff</td><td><b>28 sen/kWh ⚠️</b></td></tr><tr><td>RE Target</td><td><b>10 GW by 2030</b></td></tr><tr><td>ATAP</td><td><b>❌ Not Available</b></td></tr><tr><td>Framework</td><td><b>SET-P (2025)</b></td></tr><tr><td>Regulator</td><td><b>SEB</b></td></tr></table>"
                },
            ]

            for r in regions_map:
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=18,
                    color=r["color"],
                    fill=True,
                    fill_color=r["color"],
                    fill_opacity=0.4,
                    tooltip=r["name"],
                    popup=folium.Popup(r["popup"], max_width=220)
                ).add_to(m)
                folium.map.Marker(
                    [r["lat"], r["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:11px;font-weight:bold;color:#1A1A1A;white-space:nowrap;margin-top:22px">{r["name"]}</div>',
                        icon_size=(150, 30), icon_anchor=(75, 0)
                    )
                ).add_to(m)

            st_folium(m, width=480, height=380)

        except ImportError:
            st.info("Install folium and streamlit-folium:\npip install folium streamlit-folium")

    # ── Plotly Bar Charts ─────────────────────────────────────────────────────
    with chart_col:
        try:
            import plotly.graph_objects as go

            regions_list = ["Peninsular", "Sabah", "Sarawak"]
            bar_colors   = ["#2D8B4E", "#D69E2E", "#E53E3E"]

            for title, values, texts in [
                ("Electricity Tariff (sen/kWh)", [40, 34.5, 28], ["40 sen", "34.5 sen", "28 sen"]),
                ("LSS Quota (MW)", [2060, 150, 0], ["2,060 MW", "150 MW ⚠️", "N/A"]),
                ("Grid Reliability — SAIDI (min, lower is better)", [47.88, 244, 0], ["47.88 min", "203–286 min ⚠️", "N/A"]),
            ]:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=regions_list, y=values,
                    marker_color=bar_colors,
                    text=texts, textposition="outside",
                    hovertemplate="<b>%{x}</b><br>" + title.split("(")[0] + ": %{text}<extra></extra>"
                ))
                fig.update_layout(
                    title=dict(text=title, font=dict(size=13, color="#1A1A1A")),
                    plot_bgcolor="#F0FFF4", paper_bgcolor="#F0FFF4",
                    font=dict(color="#1A1A1A", size=12),
                    height=165, margin=dict(t=35, b=10, l=10, r=10),
                    showlegend=False,
                    yaxis=dict(showgrid=False, showticklabels=False),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.info("Install plotly:\npip install plotly")

st.divider()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            st.session_state.vectorstore = load_vectorstore()
            st.session_state.db_loaded = True
        except Exception as e:
            st.session_state.db_loaded = False
            st.session_state.db_error = str(e)

if not st.session_state.get("db_loaded", False):
    st.error(f"⚠️ Could not load knowledge base: {st.session_state.get('db_error', 'Unknown error')}")
    st.stop()

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg.get("alerts"):
            for alert in msg["alerts"]:
                st.markdown(f"""
                <div class="alert-box">
                    <div class="alert-title">⚠️ Unknown Unknown — Did You Know?</div>
                    {alert}
                </div>
                """, unsafe_allow_html=True)

        rag_sources = msg.get("rag_sources", [])
        web_sources = msg.get("web_sources", [])

        if rag_sources or web_sources:
            with st.expander("📚 Sources", expanded=False):
                if rag_sources:
                    st.markdown('<div class="source-box source-db"><div class="source-label">📁 Database Sources</div>', unsafe_allow_html=True)
                    for src in rag_sources:
                        title = src.get("title", "Unknown")
                        org   = src.get("org", "")
                        fname = src.get("filename", "")
                        st.markdown(f"• **{title}** — {org} `{fname}`")
                    st.markdown('</div>', unsafe_allow_html=True)

                if web_sources:
                    st.markdown('<div class="source-box source-web"><div class="source-label">🌐 Web Sources</div>', unsafe_allow_html=True)
                    for src in web_sources:
                        title = src.get("title", "Unknown")
                        url   = src.get("url", "")
                        st.markdown(f"• [{title}]({url})")
                    st.markdown('</div>', unsafe_allow_html=True)

        if msg.get("web_used"):
            st.caption("🌐 Live web search was used for this response")

# ── Suggested Questions ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking...")
    cols = st.columns(2)
    suggestions = [
        "What is the current solar programme in Malaysia?",
        "What is the LCOE for utility-scale solar in Malaysia?",
        "How do I apply for Solar ATAP?",
        "What are the EIA requirements for a solar farm?",
        "What is different about solar development in Sabah?",
        "What are the risks I should know about before investing?",
    ]
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, use_container_width=True, key=f"sugg_{i}"):
                st.session_state.pending_query = suggestion
                st.rerun()

# ── Chat Input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask about Malaysia solar development...")

if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

# ── Process Query ─────────────────────────────────────────────────────────────
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Searching knowledge base..."):
        try:
            result = run_ask(query, region, st.session_state.vectorstore,
                             history=st.session_state.messages)
            answer      = result.get("answer", "No answer generated.")
            rag_sources = result.get("rag_sources", [])
            web_sources = result.get("web_sources", [])
            alerts      = result.get("alerts", [])
            web_used    = result.get("web_used", False)

        except Exception as e:
            answer      = f"⚠️ Error: {str(e)}"
            rag_sources = []
            web_sources = []
            alerts      = []
            web_used    = False

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "rag_sources": rag_sources,
        "web_sources": web_sources,
        "alerts": alerts,
        "web_used": web_used
    })

    st.rerun()