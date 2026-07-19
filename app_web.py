"""
app_web.py
----------
Streamlit web dashboard for DriveWise, using the primary real-model
pipeline (src/). This satisfies the "Web interface for user interaction"
line in the problem statement's tech stack.

Chat-style interface with conversation history, plus an expandable
"Technical Sources" panel per answer showing brand, model, section,
page number, brochure file, and chunk reference - so every answer can
be traced back to an exact page.

Run with: streamlit run app_web.py
"""

import requests
import streamlit as st
from src.pipeline import DriveWisePipeline

try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False


@st.cache_resource
def get_pipeline():
    return DriveWisePipeline()


def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


st.set_page_config(page_title="DriveWise AI", page_icon="🚗", layout="centered")

# Modern glassmorphism-style CSS
st.markdown("""
<style>
    .glow-title {
        font-size: 3rem;
        font-weight: 800;
        color: #fff;
        text-align: center;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #a0aab5;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .stTextInput input:focus, .stSelectbox select:focus {
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
        border-color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-title">🚗 DriveWise AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Metadata-Aware Automotive RAG Assistant</div>', unsafe_allow_html=True)

with st.spinner("Loading models (first run downloads them from HuggingFace)..."):
    pipeline = get_pipeline()

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if HAS_LOTTIE:
        lottie_anim = load_lottie_url(
            "https://lottie.host/81b3d0b2-7bc7-4348-8ec1-cbf7384a3c10/D1q9c60e4S.json"
        )
        if lottie_anim:
            st_lottie(lottie_anim, height=150, key="ai_anim")
        else:
            st.write("🤖")
    else:
        st.write("🤖")

    st.markdown("<h3 style='text-align: center;'>🚘 Vehicle Selection</h3>", unsafe_allow_html=True)
    cars = pipeline.list_available_cars()
    car_labels = [f"{brand} {model}" for brand, model in cars]
    PLACEHOLDER = "--Select a vehicle--"
    options = [PLACEHOLDER] + car_labels
    selected_label = st.selectbox(
        "Choose a vehicle to analyze:",
        options,
        index=0,
        label_visibility="collapsed"
    )
    car_selected = selected_label != PLACEHOLDER
    if car_selected:
        selected_index = car_labels.index(selected_label)
        brand, model = cars[selected_index]
        # Clear history if the user switches to a different car
        if st.session_state.get("active_car") != selected_label:
            st.session_state.messages = []
            st.session_state["active_car"] = selected_label
    else:
        brand, model = None, None

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander("📂 View Technical Sources"):
        for s in sources:
            st.caption(
                f"**{s['brand']} {s['model']}** | Section: *{s['section']}* "
                f"| Page: {s['page_number']} | File: `{s['brochure_file']}` "
                f"| Chunk ref: `{s['chunk_reference']}` | Relevance: {s['relevance_score']}"
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if not car_selected:
    st.chat_input("Select a vehicle first...", disabled=True)
else:
    if prompt := st.chat_input("Ask a question about this car (e.g., 'What is the mileage?')"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Processing through the RAG pipeline..."):
            result = pipeline.ask(brand, model, prompt)
            answer = result["answer"]
            sources = result.get("sources", [])

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    render_sources(sources)
                st.caption(f"⚡ Answered in {result['response_time_seconds']}s")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
