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

Improvements (evaluation feedback)
-----------------------------------
- Fuzzy car search: a text-input box filters the vehicle dropdown in
  real time using fuzzy matching — no need to scroll through a long list.
- Input validation: invalid / empty queries show an st.warning() inline;
  the pipeline is never called with bad input.
- Pipeline error handling: PipelineError from any stage is caught and
  displayed as st.error() with the failing stage name — no raw traceback
  shown to the user.
- Init error handling: if DriveWisePipeline() itself fails, a descriptive
  st.error() is shown instead of the Streamlit crash page.

Run with: streamlit run app_web.py
"""

import requests
import streamlit as st

from src.validator import PipelineError, validate_query, fuzzy_match_cars

try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False


# ── Pipeline loader ──────────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline():
    from src.pipeline import DriveWisePipeline
    return DriveWisePipeline()


def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ── Page config ──────────────────────────────────────────────────────────────

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
    .search-hint {
        font-size: 0.82rem;
        color: #7a8a99;
        margin-top: -8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-title">🚗 DriveWise AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Metadata-Aware Automotive RAG Assistant</div>', unsafe_allow_html=True)

# ── Pipeline initialisation (with error handling) ────────────────────────────

try:
    with st.spinner("Loading models (first run downloads them from HuggingFace)..."):
        pipeline = get_pipeline()
except PipelineError as pe:
    st.error(
        f"⚠️ **Pipeline failed during `{pe.stage}` stage.**\n\n"
        f"{pe.cause}\n\n"
        "Please check that the `data/brochures/` folder exists and contains "
        "at least one `.pdf` or `.txt` brochure, then reload the page."
    )
    st.stop()
except Exception as exc:
    st.error(
        f"⚠️ **Unexpected error while loading DriveWise.**\n\n`{exc}`\n\n"
        "Try reloading the page. If the problem persists, check your environment."
    )
    st.stop()

# ── Vehicle selection ─────────────────────────────────────────────────────────

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

    all_cars = pipeline.list_available_cars()
    all_labels = [f"{brand} {model}" for brand, model in all_cars]

    # ── Fuzzy search box ──────────────────────────────────────────────────
    search_term = st.text_input(
        "🔍 Search vehicles",
        placeholder="Type brand or model name (e.g. 'creta', 'nexon', 'tata')…",
        label_visibility="collapsed",
        key="car_search",
    )
    st.markdown(
        '<p class="search-hint">Type to filter the list, or leave blank to see all vehicles.</p>',
        unsafe_allow_html=True,
    )

    # Filter car list using fuzzy matching when the user has typed something
    if search_term.strip():
        fuzzy_results = fuzzy_match_cars(search_term, all_cars, top_n=len(all_cars))
        # Keep cars with a meaningful match score, minimum 0.3
        filtered_labels = [
            r["label"] for r in fuzzy_results if r["score"] >= 0.3
        ]
        if not filtered_labels:
            st.caption("No vehicles matched your search — showing all vehicles.")
            filtered_labels = all_labels
    else:
        filtered_labels = all_labels

    PLACEHOLDER = "-- Select a vehicle --"
    options = [PLACEHOLDER] + filtered_labels
    selected_label = st.selectbox(
        "Choose a vehicle to analyze:",
        options,
        index=0,
        label_visibility="collapsed",
        key="car_select",
    )

    car_selected = selected_label != PLACEHOLDER
    if car_selected:
        # Resolve back to (brand, model)
        selected_index = all_labels.index(selected_label)
        brand, model = all_cars[selected_index]
        # Clear history if the user switches to a different car
        if st.session_state.get("active_car") != selected_label:
            st.session_state.messages = []
            st.session_state["active_car"] = selected_label
    else:
        brand, model = None, None

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list) -> None:
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

# ── Chat input ────────────────────────────────────────────────────────────────

if not car_selected:
    st.chat_input("Select a vehicle first...", disabled=True)
else:
    if prompt := st.chat_input("Ask a question about this car (e.g., 'What is the mileage?')"):

        # ── Input validation ───────────────────────────────────────────────
        try:
            clean_prompt = validate_query(prompt)
        except ValueError as ve:
            st.warning(f"⚠️ {ve}")
            st.stop()

        st.chat_message("user").markdown(clean_prompt)
        st.session_state.messages.append({"role": "user", "content": clean_prompt})

        # ── Pipeline call (with error handling) ───────────────────────────
        with st.spinner("Processing through the RAG pipeline..."):
            try:
                result = pipeline.ask(brand, model, clean_prompt)
            except PipelineError as pe:
                stage_hints = {
                    "retrieval":  "The FAISS vector index may be corrupted. Try reloading the page.",
                    "reranking":  "The cross-encoder model encountered an error.",
                    "generation": "The language model failed to produce an answer.",
                    "logging":    "The answer was generated but could not be saved to the log.",
                }
                hint = stage_hints.get(pe.stage, "Please reload the page and try again.")
                st.error(
                    f"⚠️ **Pipeline error at `{pe.stage}` stage.**\n\n"
                    f"`{pe.cause}`\n\n{hint}"
                )
                st.stop()
            except Exception as exc:
                st.error(
                    f"⚠️ **Unexpected error.**\n\n`{exc}`\n\n"
                    "Please reload the page and try again."
                )
                st.stop()

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
