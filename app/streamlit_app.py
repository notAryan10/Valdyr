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
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Text */
    p {
        color: #8b949e;
        font-size: 1.1rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    .stSlider > div > div > div > div {
        background-color: #238636 !important;
    }

    .stNumberInput > div > div > input {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }

    .stCheckbox > label > div[role="checkbox"] {
        background-color: #21262d;
        border: 1px solid #30363d;
    }

    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: #238636;
        border: 1px solid #238636;
    }

    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        padding: 0.5rem 1rem !important;
    }

    .stButton > button:hover {
        background-color: #2ea043 !important;
        border-color: rgba(240, 246, 252, 0.1) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Result Card */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        background: linear-gradient(145deg, #161b22, #0d1117);
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-out;
    }

    .result-card h3 {
        color: #8b949e;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-card h1 {
        color: #58a6ff;
        font-size: 3rem;
        margin: 0;
        text-shadow: 0 0 20px rgba(88, 166, 255, 0.3);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Dividers */
    hr {
        border-color: #30363d;
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
        use_container_width=True,
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
                st.balloons()
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
                
# ---------------- FOOTER ---------------- #
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ for rapid real estate prediction | Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)