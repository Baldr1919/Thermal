
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Soil Thermal Conductivity Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. PATHS AND FEATURE DEFINITIONS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR / "Data.xlsx"

FEATURES = [
    "Dry density",
    "Porosity",
    "Saturation degree",
    "Quartz content",
    "Sand content",
    "Clay content",
]

FEATURE_HELP = {
    "Dry density": "Use the same unit as the original training dataset.",
    "Porosity": "Use the same numerical scale as the training dataset.",
    "Saturation degree": "Use the same numerical scale as the training dataset.",
    "Quartz content": "Use the same scale used to develop the model and equation.",
    "Sand content": "Use the same scale as the training dataset.",
    "Clay content": "Use the same scale as the training dataset.",
}


# =========================================================
# 3. PROFESSIONAL MATPLOTLIB DEFAULTS
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "text.color": "#172033",
    "axes.labelcolor": "#172033",
    "xtick.color": "#172033",
    "ytick.color": "#172033",
})


# =========================================================
# 4. STREAMLIT CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #F3F6FA;
    --surface: #FFFFFF;
    --text: #172033;
    --muted: #475569;
    --border: #D7E0EA;
    --navy: #17324D;
    --teal: #087F8C;
    --teal-dark: #066773;
    --soft-teal: #E5F4F3;
    --soft-blue: #EAF0F8;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Inter', 'DejaVu Sans', Arial, sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'DejaVu Sans', Arial, sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: rgba(243, 246, 250, 0.96);
}

h1, h2, h3, h4, p, li, label {
    color: var(--text);
}

.hero {
    background: linear-gradient(115deg, #FFFFFF 0%, #E5F1F5 100%);
    border: 1px solid #D4E2EC;
    border-radius: 20px;
    padding: 34px;
    margin-bottom: 22px;
}

.eyebrow {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
    color: var(--teal) !important;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(28px, 4vw, 40px);
    font-weight: 800;
    line-height: 1.2;
    color: var(--navy) !important;
    margin-bottom: 14px;
}

.hero-description {
    max-width: 900px;
    font-size: 15px;
    line-height: 1.9;
    color: #334155 !important;
    margin-bottom: 0;
}

.section-heading {
    color: var(--navy) !important;
    font-size: 23px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 7px;
}

.section-description {
    font-size: 14px;
    line-height: 1.8;
    color: var(--muted) !important;
    margin-bottom: 18px;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0 20px 0;
}

.panel-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--navy) !important;
    margin-bottom: 8px;
}

.panel-text {
    font-size: 13px;
    line-height: 1.85;
    color: #334155 !important;
}

div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label {
    color: #172033 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] {
    background: white !important;
    border-radius: 9px !important;
}

div[data-baseweb="input"] input {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-top: 4px solid var(--teal);
    border-radius: 14px;
    padding: 20px 22px;
    min-height: 140px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #172033 !important;
    font-weight: 800 !important;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: var(--teal) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 700 !important;
    font-size: 15px !important;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--teal-dark) !important;
    color: white !important;
}

.formula-box {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-left: 5px solid var(--teal);
    border-radius: 12px;
    padding: 22px;
    margin: 12px 0 20px 0;
}

.small-note {
    color: #475569 !important;
    font-size: 12px;
    line-height: 1.8;
}

[data-testid="stAlert"] p {
    color: #172033 !important;
}

[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #172033 !important;
    font-weight: 600;
}

hr {
    border-color: #D7E0EA !important;
}

.footer {
    text-align: center;
    color: #475569 !important;
    font-size: 12px;
    line-height: 1.9;
    padding-top: 18px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. LOAD MODEL AND EXCEL DATA
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_excel_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_excel(DATA_PATH)

    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            "The following required columns are missing from Data.xlsx: "
            + ", ".join(missing)
        )

    # Keep the six model features and ensure numeric values
    X = df[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Rows with no feature values cannot be used as background data
    X = X.dropna(how="all").reset_index(drop=True)

    if X.empty:
        raise ValueError("No usable feature rows were found in the Excel file.")

    return X


# Optional upload, useful if the dataset is not bundled with the app.
with st.expander("Dataset source", expanded=False):
    st.write(
        "By default, the app reads Data.xlsx from the same folder as app.py. "
        "You can optionally select another Excel file with the same six feature columns."
    )
    uploaded_excel = st.file_uploader(
        "Optional Excel dataset",
        type=["xlsx"],
        help="The selected file will be used as the SHAP reference dataset."
    )

try:
    model = load_model()

    if uploaded_excel is not None:
        X_reference_raw = load_excel_data(uploaded_excel.getvalue())
    else:
        if not DATA_PATH.exists():
            st.error(
                "Data.xlsx was not found. Add Data.xlsx to the project repository "
                "beside app.py, or upload it using Dataset source above."
            )
            st.stop()

        X_reference_raw = load_excel_data()

except Exception as error:
    st.error("The model or reference dataset could not be loaded.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# =========================================================
# 6. GET THE PREPROCESSOR AND TREE ESTIMATOR
# =========================================================
try:
    # The saved PyCaret pipeline contains preprocessing steps
    # followed by GradientBoostingRegressor.
    estimator = model.steps[-1][1]
    preprocessor = model[:-1]

    if not hasattr(estimator, "feature_importances_"):
        raise TypeError(
            "The final model does not expose tree feature importances. "
            "TreeExplainer requires a compatible tree-based estimator."
        )

    def transform_features(X_raw):
        """Apply the saved pipeline preprocessing without predicting."""
        transformed = preprocessor.transform(X_raw)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        if transformed.ndim != 2 or transformed.shape[1] != len(FEATURES):
            raise ValueError(
                "The preprocessed feature dimensions do not match the six input features."
            )

        return pd.DataFrame(
            transformed,
            columns=FEATURES,
            index=X_raw.index
        )

    # SHAP background uses the actual reference dataset after preprocessing.
    X_background = transform_features(X_reference_raw)

    # Use the same transformed feature space as the underlying tree estimator.
    explainer = shap.TreeExplainer(
        estimator,
        data=X_background,
        feature_perturbation="interventional"
    )

except Exception as error:
    st.error("SHAP initialization failed.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# =========================================================
# 7. HEADER AND INTRODUCTION
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">GEOTECHNICAL ENGINEERING · RESEARCH APPLICATION</div>
    <div class="hero-title">
        Soil Thermal Conductivity<br>Prediction & Explainability
    </div>
    <p class="hero-description">
        This application estimates soil thermal conductivity using a trained
        Gradient Boosting model and a proposed analytical equation.
        SHAP analysis provides a local explanation of the Machine Learning
        prediction by showing how each input feature contributes to the
        predicted value relative to the model's reference output.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <div class="panel-title">How to use the application</div>
    <div class="panel-text">
        <b>1.</b> Enter the six soil properties using the same units and scales
        as the model's training dataset.<br>
        <b>2.</b> Run the prediction to calculate the ML estimate and the
        analytical-equation estimate.<br>
        <b>3.</b> Inspect the SHAP Waterfall plot to understand which features
        increase or decrease the ML prediction for this specific sample.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. INPUT FORM
# =========================================================
st.markdown(
    '<div class="section-heading">01 | Soil Properties</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The example values are initialized from the medians of the reference dataset. '
    'Adjust them to describe the sample you want to evaluate.'
    '</div>',
    unsafe_allow_html=True
)

defaults = {}
for col in FEATURES:
    valid_values = X_reference_raw[col].dropna()
    defaults[col] = (
        float(valid_values.median()) if not valid_values.empty else 0.0
    )

with st.form("soil_prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        dry_density = st.number_input(
            "Dry density",
            value=defaults["Dry density"],
            format="%.6f",
            help=FEATURE_HELP["Dry density"]
        )

        porosity = st.number_input(
            "Porosity",
            value=defaults["Porosity"],
            format="%.6f",
            help=FEATURE_HELP["Porosity"]
        )

    with c2:
        saturation = st.number_input(
            "Saturation degree",
            value=defaults["Saturation degree"],
            format="%.6f",
            help=FEATURE_HELP["Saturation degree"]
        )

        quartz_content = st.number_input(
            "Quartz content",
            value=defaults["Quartz content"],
            format="%.6f",
            help=FEATURE_HELP["Quartz content"]
        )

    with c3:
        sand_content = st.number_input(
            "Sand content",
            value=defaults["Sand content"],
            format="%.6f",
            help=FEATURE_HELP["Sand content"]
        )

        clay_content = st.number_input(
            "Clay content",
            value=defaults["Clay content"],
            format="%.6f",
            help=FEATURE_HELP["Clay content"]
        )

    submitted = st.form_submit_button(
        "Predict and Explain",
        use_container_width=True
    )


# =========================================================
# 9. PREDICTIONS AND SHAP WATERFALL
# =========================================================
if submitted:

    input_data = pd.DataFrame([{
        "Dry density": dry_density,
        "Porosity": porosity,
        "Saturation degree": saturation,
        "Quartz content": quartz_content,
        "Sand content": sand_content,
        "Clay content": clay_content,
    }], columns=FEATURES)

    try:
        # -------------------------------------------------
        # 9.1 ML prediction from the full saved pipeline
        # -------------------------------------------------
        ml_prediction = float(model.predict(input_data)[0])

        # -------------------------------------------------
        # 9.2 Analytical equation
        # -------------------------------------------------
        equation_prediction = (
            2.740221
            - 0.598635 * quartz_content
            + 3.078759 * saturation
            - 8.672656 * porosity
            + 1.399861 * quartz_content**2
            - 1.991327 * saturation**2
            + 6.718210 * porosity**2
        )

        # -------------------------------------------------
        # 9.3 Transform this sample using saved preprocessing
        # -------------------------------------------------
        X_sample_transformed = transform_features(input_data)

        # SHAP values explain the underlying tree estimator.
        shap_explanation = explainer(X_sample_transformed)

        st.divider()

        st.markdown(
            '<div class="section-heading">02 | Prediction Results</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2)

        with result1:
            st.markdown(
                '<div class="eyebrow">MODEL 01</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Machine Learning**")
            st.metric(
                "Predicted thermal conductivity",
                f"{ml_prediction:.6f}"
            )
            st.markdown(
                '<p class="small-note">'
                'Output from the saved Gradient Boosting pipeline.'
                '</p>',
                unsafe_allow_html=True
            )

        with result2:
            st.markdown(
                '<div class="eyebrow">MODEL 02</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Analytical Equation**")
            st.metric(
                "Predicted thermal conductivity",
                f"{equation_prediction:.6f}"
            )
            st.markdown(
                '<p class="small-note">'
                'Output from the proposed quadratic equation.'
                '</p>',
                unsafe_allow_html=True
            )

        difference = ml_prediction - equation_prediction

        st.write("")
        d1, d2 = st.columns(2)

        with d1:
            st.metric(
                "Absolute difference",
                f"{abs(difference):.6f}"
            )

        with d2:
            st.metric(
                "ML minus equation",
                f"{difference:+.6f}"
            )

        st.markdown(
            '<p class="small-note">'
            'The difference compares two predictions; it is not an error '
            'against an experimental measurement.'
            '</p>',
            unsafe_allow_html=True
        )

        with st.expander("View input values"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # 9.4 SHAP WATERFALL PLOT
        # -------------------------------------------------
        st.divider()

        st.markdown(
            '<div class="section-heading">03 | SHAP Waterfall Explanation</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="section-description">
            This plot explains the Machine Learning prediction for the
            selected sample. Features that push the prediction higher
            move it away from the baseline in the positive direction;
            features that push it lower move it in the negative direction.
        </div>
        """, unsafe_allow_html=True)

        # Explain only this new sample.
        # With interventional TreeExplainer, the baseline is based on
        # the reference dataset supplied to the explainer.
        sample_explanation = shap_explanation[0]

        fig = plt.figure(figsize=(11, 6.5), facecolor="white")

        shap.plots.waterfall(
            sample_explanation,
            max_display=len(FEATURES),
            show=False
        )

        fig = plt.gcf()
        ax = plt.gca()

        # Remove grid lines and keep the plot clean.
        ax.grid(False)
        ax.set_axisbelow(False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # Improve typography and remove unnecessary borders.
        ax.tick_params(axis="both", labelsize=10, colors="#172033")
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""
        <div class="panel">
            <div class="panel-title">How to interpret the plot</div>
            <div class="panel-text">
                <b>Baseline:</b> the reference prediction derived from the
                background data used by the SHAP explainer.<br>
                <b>Positive SHAP value:</b> a feature contribution that raises
                the prediction relative to the baseline.<br>
                <b>Negative SHAP value:</b> a feature contribution that lowers
                the prediction relative to the baseline.<br>
                <b>Final output:</b> the baseline plus all SHAP contributions,
                which should reproduce the underlying estimator's output
                up to numerical tolerance.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "The Waterfall plot explains the ML model, not the analytical equation. "
            "SHAP contributions depend on the model, sample, and reference data."
        )

    except Exception as error:
        st.error("Prediction or SHAP explanation failed.")
        with st.expander("Technical details"):
            st.code(str(error))


# =========================================================
# 10. ANALYTICAL EQUATION
# =========================================================
st.divider()

st.markdown(
    '<div class="section-heading">04 | Proposed Analytical Equation</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="section-description">
    The equation uses quartz content, saturation degree, and porosity.
    Dry density, sand content, and clay content are not included in this equation.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="formula-box">', unsafe_allow_html=True)

st.latex(
    r"""
    \begin{aligned}
    K ={}& 2.740221
    - 0.598635 Q
    + 3.078759 S
    - 8.672656 n \\[5pt]
    &+ 1.399861 Q^2
    - 1.991327 S^2
    + 6.718210 n^2
    \end{aligned}
    """
)

st.markdown("""
<div class="panel-text">
    <b>Definitions</b><br>
    <b>K</b>: predicted thermal conductivity<br>
    <b>Q</b>: quartz content<br>
    <b>S</b>: saturation degree<br>
    <b>n</b>: porosity<br><br>
    <b>Scale requirement:</b> Enter the variables on the same numerical
    scales used to develop the equation. For example, fractions and percentages
    are not interchangeable.
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 11. REFERENCE DATA AND MODEL NOTES
# =========================================================
with st.expander("Reference dataset and model details"):

    st.write(f"Reference dataset rows: {len(X_reference_raw)}")
    st.write(f"Number of input features: {len(FEATURES)}")

    st.markdown("**Feature ranges in the reference dataset**")

    range_table = pd.DataFrame({
        "Feature": FEATURES,
        "Minimum": [
            X_reference_raw[c].min() for c in FEATURES
        ],
        "Median": [
            X_reference_raw[c].median() for c in FEATURES
        ],
        "Maximum": [
            X_reference_raw[c].max() for c in FEATURES
        ],
    })

    st.dataframe(
        range_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Scientific notes**

    - Predictions outside the training-data range may be unreliable.
    - SHAP is a model-explanation method, not evidence of causality.
    - The reference dataset is used to establish the SHAP background.
    - The analytical equation uses only three of the six input features.
    - Verify the thermal conductivity unit against the original study.
    """)


# =========================================================
# 12. FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="footer">
    Soil Thermal Conductivity Prediction & Explainability<br>
    Gradient Boosting · Analytical Modeling · SHAP
</div>
""", unsafe_allow_html=True)
