import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import joblib
import traceback

from src.agent.langgraph_flow import run_advisory
from src.agent.report_pdf import generate_pdf

st.set_page_config(
    page_title="Valdýr – Housing Price Prediction & Real Estate Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #1e293b; }
    p { color: #475569; font-size: 1.05rem; }

    /* Result card */
    .result-card {
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        text-align: center;
        box-shadow: 0 4px 24px -4px rgba(37, 99, 235, 0.10), 0 2px 8px -2px rgba(0,0,0,0.04);
        margin-top: 1rem;
    }
    .result-card .label {
        color: #64748b;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .result-card .price {
        font-size: 3rem;
        font-weight: 700;
        color: #2563eb;
        line-height: 1.1;
        margin: 0.2rem 0 0.6rem;
    }
    .result-card .sub {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Advisory report report body */
    .report-body {
        background: #f8faff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        font-size: 0.97rem;
        line-height: 1.75;
        color: #334155;
    }

    hr { border-color: #e2e8f0; margin: 2rem 0; }
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }

    [data-testid="stSidebar"] { background: #f1f5f9; }

    @media (prefers-color-scheme: dark) {
        h1, h2, h3 { color: #f8fafc; }
        p { color: #cbd5e1; }
        .result-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-color: #334155;
        }
        .result-card .price { color: #60a5fa; }
        .result-card .label { color: #94a3b8; }
        .report-body { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        hr, .stDataFrame { border-color: #334155; }
        [data-testid="stSidebar"] { background: #0f172a; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    model = joblib.load(BASE_DIR / "house_price_model.pkl")
    model_columns = joblib.load(BASE_DIR / "model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error(f"**Model load failed:** {e}")
    st.stop()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🏡 Valdýr: Housing Price Prediction & Real Estate Advisor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Enter property details to get an ML-powered price estimate and an AI-generated real estate advisory report.</p>",
    unsafe_allow_html=True
)
st.divider()

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Property Features")

area         = st.sidebar.slider("📐 Area (sq ft)", 500, 10000, 2000, step=100)
bedrooms     = st.sidebar.number_input("🛏️ Bedrooms",  min_value=1, max_value=6,  value=3)
bathrooms    = st.sidebar.number_input("🛁 Bathrooms", min_value=1, max_value=5,  value=2)
stories      = st.sidebar.number_input("🏢 Stories",   min_value=1, max_value=4,  value=2)
parking      = st.sidebar.number_input("🚗 Parking Spots", min_value=0, max_value=3, value=1)

st.sidebar.markdown("### 🌟 Amenities")
mainroad       = st.sidebar.checkbox("🛣️ Near Main Road",      value=True)
guestroom      = st.sidebar.checkbox("🛌 Guest Room",           value=False)
basement       = st.sidebar.checkbox("🏚️ Basement",             value=False)
hotwaterheating= st.sidebar.checkbox("🔥 Hot Water Heating",   value=False)
airconditioning= st.sidebar.checkbox("❄️ Air Conditioning",     value=True)
prefarea       = st.sidebar.checkbox("📍 Preferred Area",       value=True)

st.sidebar.markdown("### 🪑 Furnishing Status")
furnishingstatus = st.sidebar.selectbox(
    "Furnishing",
    ["furnished", "semi-furnished", "unfurnished"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### 🤖 AI Engine")
st.sidebar.info("Gemini (1.5 Flash)")
selected_provider = "gemini"

# ─── Feature Dict & DataFrame ─────────────────────────────────────────────────
# Build the raw feature dict (same structure used by both prediction tabs)
input_dict = {
    "area":                          area,
    "bedrooms":                      bedrooms,
    "bathrooms":                     bathrooms,
    "stories":                       stories,
    "parking":                       parking,
    "mainroad":                      1 if mainroad        else 0,
    "guestroom":                     1 if guestroom       else 0,
    "basement":                      1 if basement        else 0,
    "hotwaterheating":               1 if hotwaterheating else 0,
    "airconditioning":               1 if airconditioning else 0,
    "prefarea":                      1 if prefarea        else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
    "furnishingstatus_unfurnished":    1 if furnishingstatus == "unfurnished"    else 0,
}

# Align with model feature columns (fill any missing with 0)
input_df = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)

# ─── Input fingerprint: detect when user changes any feature ──────────────────
# Used to auto-clear stale session_state results
current_input_key = str(sorted(input_dict.items()))

if st.session_state.get("_last_input_key") != current_input_key:
    st.session_state["_last_input_key"]    = current_input_key
    st.session_state["predicted_price"]    = None
    st.session_state["advisory_result"]    = None
    st.session_state["advisory_pdf"]       = None
    st.session_state["advisory_property"]  = None

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰 Price Prediction", "📋 Advisory Report"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Price Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📊 Property Feature Summary")
        display_dict = {
            "Area (sq ft)":     area,
            "Bedrooms":         bedrooms,
            "Bathrooms":        bathrooms,
            "Stories":          stories,
            "Parking":          parking,
            "Main Road":        "Yes" if mainroad         else "No",
            "Guest Room":       "Yes" if guestroom        else "No",
            "Basement":         "Yes" if basement         else "No",
            "Hot Water Heat":   "Yes" if hotwaterheating  else "No",
            "Air Conditioning": "Yes" if airconditioning  else "No",
            "Preferred Area":   "Yes" if prefarea         else "No",
            "Furnishing":       furnishingstatus.title(),
        }
        display_df = pd.DataFrame(
            [(k, str(v)) for k, v in display_dict.items()],
            columns=["Feature", "Value"]
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with col2:
        st.markdown("### 💰 Price Estimation")
        st.write("Adjust the property features using the sidebar, then click **Predict Price** to get the ML model's estimate.")

        if st.button("Predict Price 🚀", use_container_width=True, type="primary", key="btn_predict"):
            with st.spinner("Running prediction model…"):
                try:
                    prediction = model.predict(input_df)[0]
                    st.session_state["predicted_price"] = float(prediction)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Show result from session state (persists across reruns)
        if st.session_state.get("predicted_price") is not None:
            price = st.session_state["predicted_price"]
            st.success("✅ Prediction complete")
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Estimated Market Price</div>
                <div class="price">₹ {int(price):,}</div>
                <div class="sub">Based on ML model trained on housing dataset</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Click **Predict Price** to see the estimated value for this configuration.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Advisory Report
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🤖 AI-Powered Real Estate Advisory Report")
    st.write(
        "Our Gemini-powered AI agent (LangGraph + RAG) will analyze your property, "
        "predict its price, retrieve relevant market knowledge, and generate a structured advisory report."
    )

    if st.button("Generate Advisory Report 📄", use_container_width=True, type="primary", key="btn_advisory"):
        with st.spinner("Running AI advisory agent… this may take 15–30 seconds"):
            try:
                property_data = {
                    "area":            area,
                    "bedrooms":        bedrooms,
                    "bathrooms":       bathrooms,
                    "stories":         stories,
                    "parking":         parking,
                    "mainroad":        1 if mainroad         else 0,
                    "guestroom":       1 if guestroom        else 0,
                    "basement":        1 if basement         else 0,
                    "hotwaterheating": 1 if hotwaterheating  else 0,
                    "airconditioning": 1 if airconditioning  else 0,
                    "prefarea":        1 if prefarea         else 0,
                    "furnishingstatus": furnishingstatus,
                }

                result = run_advisory(property_data, provider=selected_provider)
                st.session_state["advisory_result"]   = result
                st.session_state["advisory_property"] = property_data

                # Generate PDF (non-blocking – skip if it fails)
                try:
                    st.session_state["advisory_pdf"] = generate_pdf(
                        result["report"], result["predicted_price"], property_data
                    )
                except Exception as pdf_err:
                    st.session_state["advisory_pdf"] = None
                    st.warning(f"PDF generation skipped: {pdf_err}")

                st.success("✅ Advisory report generated successfully!")

            except Exception as e:
                st.error(f"**Report generation failed:** {e}")
                with st.expander("🔍 Error details (for debugging)"):
                    st.code(traceback.format_exc())

    # ── Display saved advisory result ─────────────────────────────────────────
    result = st.session_state.get("advisory_result")
    if result:
        predicted = result.get("predicted_price", 0)
        st.markdown(f"""
        <div class="result-card">
            <div class="label">AI-Predicted Property Price</div>
            <div class="price">₹ {int(predicted):,}</div>
            <div class="sub">Model estimate used for advisory analysis</div>
        </div>
        """, unsafe_allow_html=True)

        if result.get("warnings"):
            st.warning(f"⚠️ Data Warnings\n\n{result['warnings']}")

        st.markdown("---")
        st.markdown("#### 📑 Advisory Report")

        # Render the report nicely
        st.markdown('<div class="report-body">', unsafe_allow_html=True)
        st.markdown(result["report"])
        st.markdown('</div>', unsafe_allow_html=True)

        # PDF download
        pdf_bytes = st.session_state.get("advisory_pdf")
        if pdf_bytes:
            st.markdown("---")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="valdyr_advisory_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("👆 Click **Generate Advisory Report** to run the full AI analysis pipeline.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "🏡 Valdýr &nbsp;|&nbsp; ML Price Prediction · LangGraph Agent · RAG Knowledge Base<br>"
    "Built with Streamlit · scikit-learn · LangGraph · FAISS · Gemini API"
    "</p>",
    unsafe_allow_html=True
)
