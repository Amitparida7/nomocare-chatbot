import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NomoCare | Mental Health Chatbot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.hero-sub {
    color: #a89ed0;
    text-align: center;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}

.chat-bubble-bot {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e8e2f8;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
}

.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}

.badge-nomophobia {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.badge-yes { background: #ff6b6b33; color: #ff6b6b; border: 1px solid #ff6b6b55; }
.badge-no  { background: #51cf6633; color: #51cf66; border: 1px solid #51cf6655; }
.badge-high { background: #ff922b33; color: #ff922b; border: 1px solid #ff922b55; }
.badge-low  { background: #74c0fc33; color: #74c0fc; border: 1px solid #74c0fc55; }

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 14px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input::placeholder {
    color: #7a6fa8 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.4) !important;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}

.sidebar-info {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px;
    font-size: 0.85rem;
    color: #a89ed0;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class MultiTaskBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.nomophobia_head = nn.Linear(768, 2)
        self.severity_head = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.nomophobia_head(pooled), self.severity_head(pooled)


# ─────────────────────────────────────────────
# KNOWLEDGE BASE & RAG
# ─────────────────────────────────────────────
knowledge_base = {
    "nomophobia": {
        "definition": "Nomophobia is the fear or anxiety of being without access to a mobile phone.",
        "symptoms": [
            "Feeling anxious without phone",
            "Constantly checking phone",
            "Fear of losing connectivity"
        ],
        "effects": [
            "Stress and anxiety",
            "Sleep disturbance",
            "Reduced focus"
        ]
    },
    "low": {
        "description": "Low severity indicates mild phone dependence.",
        "recommendation": [
            "Limit screen time",
            "Use phone only when needed"
        ]
    },
    "elevated": {
        "description": "Elevated severity indicates strong phone dependence.",
        "recommendation": [
            "Practice digital detox",
            "Reduce notifications",
            "Engage in offline activities"
        ]
    }
}

def build_documents():
    docs = []
    nomo = knowledge_base["nomophobia"]
    docs.append(nomo["definition"])
    for s in nomo["symptoms"]:
        docs.append("Behavior: " + s)
    for e in nomo["effects"]:
        docs.append("Impact: " + e)
    for key in ["low", "elevated"]:
        docs.append(knowledge_base[key]["description"])
        for r in knowledge_base[key]["recommendation"]:
            docs.append("Advice: " + r)
    return docs


# ─────────────────────────────────────────────
# CACHED RESOURCE LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_bert_model():
    import gdown
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskBERT().to(device)

    model_path = "best_model_fixed.pt"

    # ── Download from Google Drive if not already present ──
    if not os.path.exists(model_path):
        GDRIVE_FILE_ID = "1A9_ug1HMqDYpFeAbKOqBZwK0HKdgb31h"
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        with st.spinner("⬇️ Downloading model from Google Drive... (one time only, ~2 min)"):
            gdown.download(url, model_path, quiet=False)

    if not os.path.exists(model_path):
        st.error("❌ Model download failed. Check your Google Drive link is public.")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, device


@st.cache_resource
def load_rag():
    documents = build_documents()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embed_model.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))
    return embed_model, index, documents


# ─────────────────────────────────────────────
# PREDICTION  (matches notebook exactly)
# ─────────────────────────────────────────────
def predict(text, model, tokenizer, device, threshold=0.6):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        nomo_out, sev_out = model(input_ids, attention_mask)

        # ── Nomophobia ──────────────────────────
        nomo_probs = torch.softmax(nomo_out, dim=1)
        nomo_pred  = torch.argmax(nomo_probs, dim=1).item()
        nomo_conf  = nomo_probs[:, 1].item()

        # ── Severity (raw) ──────────────────────
        sev_probs  = torch.softmax(sev_out, dim=1)[0]
        sev_pred   = torch.argmax(sev_probs).item()

        # ── Smart keyword override (from notebook) ──
        strong_keywords   = ["panic", "constantly", "addicted",
                             "cannot", "lost without", "extremely"]
        moderate_keywords = ["anxious", "checking", "again and again", "stress"]
        text_lower = text.lower()

        if any(w in text_lower for w in strong_keywords):
            sev_pred = 1                           # force Elevated
        elif any(w in text_lower for w in moderate_keywords):
            if sev_probs[1].item() > 0.20:
                sev_pred = 1                       # moderate → Elevated

    # ── Labels (notebook mapping: 0=Low, 1=Elevated) ──
    nomo_label = "Nomophobia"    if nomo_pred == 1 else "No Nomophobia"
    sev_label  = "Elevated"      if sev_pred  == 1 else "Low"

    return nomo_label, sev_label, nomo_conf


# ─────────────────────────────────────────────
# RAG RETRIEVAL
# ─────────────────────────────────────────────
def retrieve_context(nomo, severity, embed_model, index, documents):
    query = f"Nomophobia: {nomo}, Severity: {severity}"
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), 3)
    results = [documents[i] for i in indices[0]]
    return "\n".join(results)


# ─────────────────────────────────────────────
# LLM RESPONSE (GROQ)
# ─────────────────────────────────────────────
def generate_response(user_text, nomo, severity, context, groq_api_key):
    client = Groq(api_key=groq_api_key)
    prompt = f"""You are a compassionate mental health assistant specialized in nomophobia (smartphone addiction/anxiety).

User Input:
{user_text}

Prediction:
Nomophobia Detected: {nomo}
Severity: {severity}

Relevant Knowledge:
{context}

Instructions:
- Respond warmly and empathetically in 3-4 sentences
- Explain what the prediction means in simple human language
- Mention specific behaviors naturally (don't use bullet points)
- Give 1-2 practical, personalized tips based on severity
- End with encouragement

Generate your response:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_key,
        placeholder="gsk_..."
    )
    if groq_key_input:
        st.session_state.groq_key = groq_key_input

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-info'>
    <b>🧠 About NomoCare</b><br><br>
    This AI chatbot detects <b>Nomophobia</b> (fear of being without a smartphone) from how you describe your phone habits.<br><br>
    <b>How it works:</b><br>
    1. You type how you feel about your phone<br>
    2. BERT model detects nomophobia + severity<br>
    3. RAG retrieves relevant advice<br>
    4. Llama 3.1 generates a caring response
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("<div class='hero-title'>🧠 NomoCare</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>LLM-Based Mental Health Chatbot · Nomophobia Detection</div>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS (with spinner)
# ─────────────────────────────────────────────
with st.spinner("Loading AI models... (first load takes ~30 seconds)"):
    bert_model, tokenizer, device = load_bert_model()
    embed_model, faiss_index, documents = load_rag()


# ─────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class='chat-bubble-bot'>
    👋 Hello! I'm NomoCare, your mental health assistant.<br><br>
    Tell me how you feel about your smartphone — like <i>"I feel anxious when my battery is low"</i> or <i>"I check my phone every few minutes"</i> — and I'll analyze your nomophobia level and share some helpful advice.
    </div>
    """, unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble-user'>💬 {entry['user']}</div>", unsafe_allow_html=True)

    nomo = entry['nomo']
    sev = entry['severity']
    conf = entry['confidence']

    nomo_badge_class = "badge-yes" if nomo == "Nomophobia" else "badge-no"
    sev_badge_class  = "badge-high" if sev == "Elevated" else "badge-low"

    st.markdown(f"""
    <div class='result-card'>
        <span style='color:#a89ed0; font-size:0.8rem; font-weight:500;'>ANALYSIS RESULT</span><br><br>
        <span style='color:#e8e2f8;'>Nomophobia Detected:</span>&nbsp;
        <span class='badge-nomophobia {nomo_badge_class}'>{nomo}</span>
        &nbsp;&nbsp;
        <span style='color:#e8e2f8;'>Severity:</span>&nbsp;
        <span class='badge-nomophobia {sev_badge_class}'>{sev}</span>
        &nbsp;&nbsp;
        <span style='color:#7a6fa8; font-size:0.82rem;'>Confidence: {conf:.0%}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='chat-bubble-bot'>🤖 {entry['bot']}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Describe how you feel about your phone usage...",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("🔍 Analyze & Respond")

if submitted and user_input.strip():
    if not st.session_state.groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar to enable chatbot responses.")
    else:
        with st.spinner("Analyzing your response..."):
            nomo, severity, confidence = predict(user_input, bert_model, tokenizer, device)
            context = retrieve_context(nomo, severity, embed_model, faiss_index, documents)
            bot_reply = generate_response(user_input, nomo, severity, context, st.session_state.groq_key)

            st.session_state.chat_history.append({
                "user": user_input,
                "nomo": nomo,
                "severity": severity,
                "confidence": confidence,
                "bot": bot_reply
            })
        st.rerun()

elif submitted and not user_input.strip():
    st.warning("Please type something before submitting.")