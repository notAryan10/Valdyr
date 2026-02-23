import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Valdýr - House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1e293b;
    }

    /* Text */
    p {
        color: #475569;
        font-size: 1.1rem;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Result Card */
    .result-card {
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    .result-card h3 {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .result-card h1 {
        font-size: 3.5rem;
        margin: 0;
        color: #2563eb;
        font-weight: 700;
    }

    /* Dividers */
    hr {
        border-color: #e2e8f0;
        margin: 2.5rem 0;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 { color: #f8fafc; }
        p { color: #cbd5e1; }
        .result-card {
            background-color: #1e293b;
            border-color: #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .result-card h1 { color: #60a5fa; }
        .result-card h3 { color: #94a3b8; }
        hr, .stDataFrame { border-color: #334155; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading House Price Prediction Model...")
def load_model():
    model_path = BASE_DIR / "house_price_model.pkl"
    columns_path = BASE_DIR / "model_columns.pkl"
    
    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError("Model files not found. Please ensure house_price_model.pkl and model_columns.pkl are in the app/ directory.")
        
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error(f"❌ Error setting up prediction model: {e}")
    st.info("💡 Make sure you have trained the model and placed the .pkl files inside the `/app` folder.")
    st.stop()

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align: center;'>🏡 Valdýr: House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get an instant estimate for your property value using Machine Learning.</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.markdown("## ⚙️ House Features")
st.sidebar.markdown("Adjust the parameters to see the predicted price.")

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

st.sidebar.markdown("### 🛋️ Furnishing Status")
furnishingstatus = st.sidebar.selectbox(
    "Select Furnishing",
    ["furnished", "semi-furnished", "unfurnished"],
    label_visibility="collapsed"
)

# ---------------- DATAFRAME ---------------- #
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

# Essential: align columns with identically to what the model expects
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ---------------- MAIN CONTENT ---------------- #
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Your Inputs")
    st.dataframe(
        pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}),
        width='stretch',
        height=400
    )

with col2:
    st.markdown("### 💰 Valuation")
    st.write("Click the button below to predict the current market price of the specified house.")
    
    if st.button("Predict Real Estate Price 🚀", use_container_width=True, type="primary"):
        with st.spinner("Analyzing market parameters..."):
            try:
                prediction = model.predict(input_df)[0]
                st.success("Analysis Complete!")
                st.markdown(f"""
                <div class="result-card">
                    <h3>Estimated Price</h3>
                    <h1>₹ {int(prediction):,}</h1>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
                
# ---------------- FOOTER ---------------- #
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ for rapid real estate prediction | Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)