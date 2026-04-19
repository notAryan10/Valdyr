import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import joblib

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

st.markdown(
    "<h1 style='text-align:center;'>🏡 Valdýr: Housing Price Prediction & Real Estate Advisor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Enter property details to get an ML-powered price estimate and an AI-generated real estate advisory report.</p>",
    unsafe_allow_html=True
)
st.divider()

st.sidebar.markdown("## ⚙️ Property Features")

area = st.sidebar.slider("📐 Area (sq ft)", 500, 10000, 2000, step=100)
bedrooms = st.sidebar.number_input("🛏️ Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.number_input("🛁 Bathrooms", 1, 5, 2)
stories = st.sidebar.number_input("🏢 Stories", 1, 4, 2)
parking = st.sidebar.number_input("🚗 Parking (Capacity)", 0, 3, 1)

st.sidebar.markdown("### 🌟 Amenities")
mainroad = st.sidebar.checkbox("🛣️ Near Main Road", value=True)
guestroom = st.sidebar.checkbox("🛌 Guest Room", value=False)
basement = st.sidebar.checkbox("🏚️ Basement", value=False)
hotwaterheating = st.sidebar.checkbox("🔥 Hot Water Heating", value=False)
airconditioning = st.sidebar.checkbox("❄️ Air Conditioning", value=True)
prefarea = st.sidebar.checkbox("📍 Preferred Area", value=True)

furnishingstatus = st.sidebar.selectbox(
    "Select Furnishing",
    ["furnished", "semi-furnished", "unfurnished"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### 🤖 AI Model")
ai_provider = st.sidebar.selectbox(
    "Select AI Engine",
    ["Groq (Llama 3.3)", "Gemini (1.5 Flash)"],
    label_visibility="collapsed"
)
provider_map = {"Groq (Llama 3.3)": "groq", "Gemini (1.5 Flash)": "gemini"}
selected_provider = provider_map[ai_provider]

input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": 1 if mainroad else 0,
    "guestroom": 1 if guestroom else 0,
    "basement": 1 if basement else 0,
    "hotwaterheating": 1 if hotwaterheating else 0,
    "airconditioning": 1 if airconditioning else 0,
    "prefarea": 1 if prefarea else 0,
    "furnishingstatus_semi-furnished": 1 if furnishingstatus == "semi-furnished" else 0,
    "furnishingstatus_unfurnished": 1 if furnishingstatus == "unfurnished" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

tab1, tab2 = st.tabs(["💰 Price Prediction", "📋 Advisory Report"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Your Inputs")
        st.dataframe(
            pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}),
            use_container_width=True,
            height=400
        )

    with col2:
        st.markdown("### 💰 Valuation")
        st.write("Click below to predict the current market price.")

        if st.button("Predict Price 🚀", width="stretch", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    prediction = model.predict(input_df)[0]
                    st.success("Done!")
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>Estimated Price</h3>
                        <h1>₹ {int(prediction):,}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.markdown("### AI Advisory Report")
    st.write("Generate a detailed advisory report with price analysis, market insights and recommendations.")

    if "advisory_result" not in st.session_state:
        st.session_state.advisory_result = None
        st.session_state.advisory_pdf = None
        st.session_state.advisory_property = None

    if st.button("Generate Advisory Report", width="stretch", type="primary"):
        with st.spinner("Running AI agent... this may take a moment"):
            try:
                property_data = {
                    "area": area,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "stories": stories,
                    "parking": parking,
                    "mainroad": 1 if mainroad else 0,
                    "guestroom": 1 if guestroom else 0,
                    "basement": 1 if basement else 0,
                    "hotwaterheating": 1 if hotwaterheating else 0,
                    "airconditioning": 1 if airconditioning else 0,
                    "prefarea": 1 if prefarea else 0,
                    "furnishingstatus": furnishingstatus,
                }

                result = run_advisory(property_data, provider=selected_provider)
                st.session_state.advisory_result = result
                st.session_state.advisory_property = property_data

                try:
                    st.session_state.advisory_pdf = generate_pdf(
                        result["report"], result["predicted_price"], property_data
                    )
                except Exception:
                    st.session_state.advisory_pdf = None

            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")

    if st.session_state.advisory_result:
        result = st.session_state.advisory_result

        st.markdown(f"""
        <div class="result-card">
            <h3>Predicted Price</h3>
            <h1>Rs {int(result['predicted_price']):,}</h1>
        </div>
        """, unsafe_allow_html=True)

        if result["warnings"]:
            st.warning(result["warnings"])

        st.markdown("---")
        st.markdown(result["report"])

        if st.session_state.advisory_pdf:
            st.download_button(
                "Download PDF Report",
                data=st.session_state.advisory_pdf,
                file_name="valdyr_advisory_report.pdf",
                mime="application/pdf",
                width="stretch"
            )

st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ | Streamlit + scikit-learn + LangGraph</p>",
    unsafe_allow_html=True
)
