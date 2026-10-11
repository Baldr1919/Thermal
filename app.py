
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Soil Thermal Conductivity | Prediction Tool",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. CUSTOM DESIGN AND TYPOGRAPHY
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #F4F7FA;
    --surface: #FFFFFF;
    --text: #111827;
    --muted: #4B5563;
    --border: #D9E2EC;
    --primary: #087F8C;
    --primary-dark: #066773;
    --soft-teal: #E6F5F5;
    --soft-blue: #EAF1FA;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', Arial, sans-serif;
    color: var(--text) !important;
}

.stApp {
    background: var(--background);
}

[data-testid="stHeader"] {
    background: rgba(244, 247, 250, 0.95);
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, li, label, span, div {
    color: var(--text);
}

.hero {
    background: linear-gradient(120deg, #FFFFFF 0%, #EAF5F7 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 34px 36px;
    margin-bottom: 24px;
}

.eyebrow {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--primary) !important;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(27px, 4vw, 39px);
    font-weight: 800;
    line-height: 1.2;
    color: #111827 !important;
    margin: 0 0 12px 0;
}

.hero-description {
    font-size: 15px;
    line-height: 1.8;
    color: #374151 !important;
    max-width: 800px;
    margin: 0;
}

.section-heading {
    font-size: 23px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 6px;
    color: #111827 !important;
}

.section-description {
    color: #4B5563 !important;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 18px;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 10px 0 18px 0;
}

.info-card h4 {
    font-size: 16px;
    font-weight: 700;
    margin: 0 0 10px 0;
    color: #111827 !important;
}

.info-card p, .info-card li {
    font-size: 13px;
    line-height: 1.8;
    color: #374151 !important;
}

.info-card ul {
    padding-left: 20px;
    margin-bottom: 0;
}

.input-panel {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 16px;
    padding: 18px 22px 8px 22px;
}

div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

input, textarea {
    color: #111827 !important;
}

div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 700 !important;
    font-size: 15px !important;
    transition: background 0.2s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--primary-dark) !important;
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 14px;
    padding: 22px 24px;
    min-height: 145px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #374151 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #111827 !important;
    font-weight: 800 !important;
}

.result-caption {
    color: #4B5563 !important;
    font-size: 12px;
    margin-top: 8px;
    line-height: 1.7;
}

.formula-box {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-left: 5px solid #087F8C;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 12px 0 18px 0;
}

.formula-note {
    font-size: 13px;
    line-height: 1.8;
    color: #374151 !important;
    margin-top: 12px;
}

div[data-testid="stAlert"] p {
    color: #111827 !important;
}

hr {
    border-color: #D9E2EC !important;
}

.footer {
    text-align: center;
    padding: 18px 0 4px 0;
    color: #4B5563 !important;
    font-size: 12px;
    line-height: 1.8;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #111827 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD THE TRAINED MACHINE LEARNING MODEL
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(
        "The trained model could not be loaded. "
        "Please check that best_model.pkl is in the same folder as app.py."
    )
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# =========================================================
# 4. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">GEOTECHNICAL ENGINEERING · RESEARCH TOOL</div>
    <div class="hero-title">Soil Thermal Conductivity<br>Prediction Tool</div>
    <p class="hero-description">
        Estimate soil thermal conductivity using two approaches:
        a trained Machine Learning model and a proposed analytical equation.
        Enter the soil properties, run both predictions, and compare the results.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. USER GUIDANCE
# =========================================================
st.markdown("""
<div class="info-card">
    <h4>How to use this tool</h4>
    <ul>
        <li>Enter all six soil properties in the input panel below.</li>
        <li>Use the same units and numerical scales as the original model training dataset.</li>
        <li>Click <b>Predict Thermal Conductivity</b> to calculate both estimates.</li>
        <li>Compare the results. Differences between the two methods are not necessarily errors.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. INPUT FORM
# =========================================================
st.markdown(
    '<div class="section-heading">01 | Soil Properties</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">'
    'Provide the values of the six input parameters. '
    'Example starting values are provided; replace them with your actual data.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("soil_input_form"):

    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        dry_density = st.number_input(
            "Dry density",
            min_value=0.0,
            value=1.60,
            step=0.05,
            format="%.4f",
            help="Enter dry density using the unit used in the training dataset."
        )

        porosity = st.number_input(
            "Porosity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            format="%.4f",
            help="This interface assumes porosity is a fraction between 0 and 1."
        )

    with col2:
        saturation = st.number_input(
            "Saturation degree",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%.4f",
            help="This interface assumes saturation degree is a fraction between 0 and 1."
        )

        quartz_content = st.number_input(
            "Quartz content",
            min_value=0.0,
            value=0.30,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the dataset used to develop the equation and ML model."
        )

    with col3:
        sand_content = st.number_input(
            "Sand content",
            min_value=0.0,
            value=0.45,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the original training data."
        )

        clay_content = st.number_input(
            "Clay content",
            min_value=0.0,
            value=0.25,
            step=0.05,
            format="%.4f",
            help="Use the same scale as the original training data."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    submitted = st.form_submit_button(
        "Predict Thermal Conductivity",
        use_container_width=True
    )


# =========================================================
# 7. PREDICTION AND RESULTS
# =========================================================
if submitted:

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

        difference = ml_prediction - equation_prediction
        absolute_difference = abs(difference)

        st.divider()

        st.markdown(
            '<div class="section-heading">02 | Prediction Results</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-description">'
            'The following results are calculated from the same input values.'
            '</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(
                '<div class="eyebrow">MODEL 01</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Machine Learning Model**")
            st.metric(
                label="Predicted thermal conductivity",
                value=f"{ml_prediction:.6f}"
            )
            st.markdown(
                '<div class="result-caption">'
                'Prediction from the saved PyCaret model '
                '(<i>Gradient Boosting Regressor</i>).'
                '</div>',
                unsafe_allow_html=True
            )

        with result_col2:
            st.markdown(
                '<div class="eyebrow">MODEL 02</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Analytical Equation**")
            st.metric(
                label="Predicted thermal conductivity",
                value=f"{equation_prediction:.6f}"
            )
            st.markdown(
                '<div class="result-caption">'
                'Prediction from the proposed quadratic equation.'
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        st.markdown("**Difference between predictions**")

        diff_col1, diff_col2 = st.columns(2)

        with diff_col1:
            st.metric(
                label="Absolute difference",
                value=f"{absolute_difference:.6f}"
            )

        with diff_col2:
            st.metric(
                label="ML minus equation",
                value=f"{difference:+.6f}"
            )

        st.caption(
            "The difference is a direct numerical comparison, not an error metric "
            "against experimental measurements."
        )

        with st.expander("View input values used for prediction"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error("Prediction failed. Please check the input values and model.")
        with st.expander("Technical details"):
            st.code(str(error))


# =========================================================
# 8. ANALYTICAL EQUATION
# =========================================================
st.divider()

st.markdown(
    '<div class="section-heading">03 | Proposed Analytical Equation</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="section-description">
    The analytical approach uses a quadratic equation based on three soil properties:
    quartz content, saturation degree, and porosity.
    The other three input parameters are used by the Machine Learning model only.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="formula-box">', unsafe_allow_html=True)

st.latex(
    r"""
    \begin{aligned}
    K ={}& 2.740221
    - 0.598635\,Q
    + 3.078759\,S
    - 8.672656\,n \\[4pt]
    &+ 1.399861\,Q^2
    - 1.991327\,S^2
    + 6.718210\,n^2
    \end{aligned}
    """
)

st.markdown("""
<div class="formula-note">
    <b>Where:</b><br>
    <b>K</b> = predicted thermal conductivity<br>
    <b>Q</b> = quartz content<br>
    <b>S</b> = saturation degree<br>
    <b>n</b> = porosity<br><br>
    <b>Important:</b> Use the same variable scales used when the equation was developed.
    The unit of K depends on the unit used in the original equation.
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 9. INPUT SCALE AND MODEL NOTES
# =========================================================
with st.expander("Input scales and scientific notes"):

    st.markdown("""
    **Machine Learning model**

    - Uses all six input variables.
    - The feature names and order are matched to the saved model pipeline.
    - Predictions may be unreliable for inputs outside the training data range.

    **Analytical equation**

    - Uses only quartz content, saturation degree, and porosity.
    - Dry density, sand content, and clay content do not affect its prediction.
    - Do not mix fractions (e.g. 0.30) and percentages (e.g. 30).

    **Units**

    - Dry density must use the training dataset's unit.
    - Confirm the exact unit of thermal conductivity from the original study.
    - Example input values are illustrative and are not validated reference cases.
    """)


# =========================================================
# 10. FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Soil Thermal Conductivity Prediction Tool<br>
    Machine Learning & Analytical Modeling
</div>
""", unsafe_allow_html=True)
