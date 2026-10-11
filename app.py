
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. Page configuration
# ==========================================
st.set_page_config(
    page_title="Soil Thermal Conductivity Predictor",
    page_icon="🌱",
    layout="centered"
)

# ==========================================
# 2. Simple styling
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    .main-title {
        text-align: center;
        color: #17324D;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 15px;
        margin-bottom: 28px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #167D8D;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #126574;
        color: white;
    }

    .section-title {
        color: #17324D;
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Load trained ML model
# ==========================================
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# ==========================================
# 4. Header
# ==========================================
st.markdown(
    '<div class="main-title">Soil Thermal Conductivity Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict thermal conductivity using Machine Learning '
    'and an Analytical Equation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# 5. Input variables
# ==========================================
st.markdown(
    '<div class="section-title">Soil Properties</div>',
    unsafe_allow_html=True
)

st.write("Enter the six input parameters below.")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        dry_density = st.number_input(
            "Dry density",
            min_value=0.0,
            value=1.60,
            step=0.05,
            format="%.4f",
            help="Use the same unit as the training dataset."
        )

        saturation = st.number_input(
            "Saturation degree",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%.4f",
            help="Assumes saturation is expressed as a fraction from 0 to 1."
        )

        sand_content = st.number_input(
            "Sand content",
            min_value=0.0,
            value=0.45,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the training dataset."
        )

    with col2:
        porosity = st.number_input(
            "Porosity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            format="%.4f",
            help="Assumes porosity is expressed as a fraction from 0 to 1."
        )

        quartz_content = st.number_input(
            "Quartz content",
            min_value=0.0,
            value=0.30,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the training dataset and equation."
        )

        clay_content = st.number_input(
            "Clay content",
            min_value=0.0,
            value=0.25,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the training dataset."
        )

    submitted = st.form_submit_button("Predict Thermal Conductivity")

# ==========================================
# 6. Prediction
# ==========================================
if submitted:

    # Keep feature names and order consistent with model training
    input_data = pd.DataFrame([{
        "Dry density": dry_density,
        "Porosity": porosity,
        "Saturation degree": saturation,
        "Quartz content": quartz_content,
        "Sand content": sand_content,
        "Clay content": clay_content
    }])

    try:
        # Machine Learning prediction
        ml_prediction = float(model.predict(input_data)[0])

        # Analytical equation prediction
        equation_prediction = (
            2.740221
            - 0.598635 * quartz_content
            + 3.078759 * saturation
            - 8.672656 * porosity
            + 1.399861 * quartz_content**2
            - 1.991327 * saturation**2
            + 6.718210 * porosity**2
        )

        # Display results
        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                label="Machine Learning Model",
                value=f"{ml_prediction:.6f}"
            )

        with result2:
            st.metric(
                label="Analytical Equation",
                value=f"{equation_prediction:.6f}"
            )

        st.caption(
            "Both values are predictions. Confirm the thermal "
            "conductivity unit used in your original dataset."
        )

        # Show input summary
        with st.expander("View input parameters"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==========================================
# 7. Footer
# ==========================================
st.divider()

st.caption(
    "Soil Thermal Conductivity Prediction | "
    "Machine Learning & Analytical Modeling"
)