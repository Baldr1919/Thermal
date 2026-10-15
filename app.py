
from pathlib import Path
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Soil Thermal Conductivity Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FILE PATHS AND FEATURES
# ============================================================
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


# ============================================================
# CUSTOM CSS
# IMPORTANT: Keep all CSS inside st.markdown()
# ============================================================
st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main header */
    .hero {
        text-align: center;
        padding: 20px 10px 25px 10px;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: #111827;
        font-size: 2.2rem;
        font-weight: 750;
        line-height: 1.3;
        margin: 0 0 12px 0;
    }

    .hero-description {
        color: #4b5563;
        font-size: 1rem;
        line-height: 1.8;
        margin: 0;
    }

    .hero-subtitle {
        color: #2563eb;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 12px;
        letter-spacing: 0.3px;
    }

    /* Guide panel */
    .guide-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 10px 0 25px 0;
        color: #111827;
    }

    .guide-card h3 {
        color: #111827;
        font-size: 1.15rem;
        margin: 0 0 15px 0;
    }

    .guide-card p {
        color: #374151;
        font-size: 0.96rem;
        line-height: 1.8;
        margin: 8px 0;
    }

    /* Section headings */
    .section-title {
        color: #111827;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    /* Input fields */
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border-color: #d1d5db !important;
    }

    div[data-testid="stNumberInput"] label {
        color: #111827 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stNumberInput"] button {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* General text */
    .stMarkdown,
    .stText,
    p,
    label {
        color: #111827;
    }

    /* Result cards */
    .result-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        min-height: 135px;
    }

    .result-label {
        color: #4b5563;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }

    .result-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 8px;
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid #e5e7eb;
        margin-top: 45px;
        padding: 25px 10px 10px 10px;
        text-align: center;
    }

    .footer-title {
        color: #111827;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .footer-affiliation {
        color: #4b5563;
        font-size: 0.9rem;
        line-height: 1.8;
        margin-bottom: 8px;
    }

    .footer-copyright {
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.7;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>Soil Thermal Conductivity Prediction</h1>

        <p class="hero-description">
            Predict soil thermal conductivity using a trained machine
            learning model and an analytical equation, with SHAP-based
            explanations of the model predictions.
        </p>

        <p class="hero-subtitle">
            Gradient Boosting &nbsp;·&nbsp;
            Analytical Modeling &nbsp;·&nbsp;
            SHAP Explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# USER GUIDE
# ============================================================
st.markdown(
    """
    <div class="guide-card">
        <h3>How to use this application</h3>

        <p>
            <strong>Step 1.</strong>
            Enter the six soil properties below. Use the same units
            and numerical scales as the original training dataset.
            Brief statistical information about the reference dataset
            is available at the end of this application.
        </p>

        <p>
            <strong>Step 2.</strong>
            Click the prediction button to calculate thermal conductivity
            using both the trained machine learning model and the
            analytical equation.
        </p>

        <p>
            <strong>Step 3.</strong>
            Review the SHAP waterfall plot to understand how each input
            contributes to the machine learning prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD REFERENCE DATASET
# ============================================================
@st.cache_data
def load_reference_data(file_bytes=None):
    if file_bytes is not None:
        return pd.read_excel(BytesIO(file_bytes))

    if DATA_PATH.exists():
        return pd.read_excel(DATA_PATH)

    return None


# Optional dataset upload
with st.expander("Dataset settings", expanded=False):
    st.write(
        "By default, the application looks for Data.xlsx in the same "
        "folder as app.py. You can upload another Excel dataset below."
    )

    uploaded_file = st.file_uploader(
        "Upload reference dataset",
        type=["xlsx", "xls"],
        key="dataset_upload",
    )

    if uploaded_file is not None:
        try:
            df = load_reference_data(uploaded_file.getvalue())
            st.success("Reference dataset loaded.")
        except Exception as e:
            df = None
            st.error(f"Could not read the uploaded dataset: {e}")
    else:
        try:
            df = load_reference_data()
        except Exception as e:
            df = None
            st.error(f"Could not read Data.xlsx: {e}")

if df is None:
    st.warning(
        "Data.xlsx was not found. Place Data.xlsx in the same folder "
        "as app.py, or upload the file using Dataset settings."
    )
    st.stop()


# ============================================================
# CHECK DATASET
# ============================================================
missing_features = [col for col in FEATURES if col not in df.columns]

if missing_features:
    st.error(
        "The reference dataset is missing these required columns: "
        + ", ".join(missing_features)
    )
    st.write("Columns found in the dataset:", df.columns.tolist())
    st.stop()

# Convert required feature columns to numeric
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES).reset_index(drop=True)

if df.empty:
    st.error("No valid rows were found in the reference dataset.")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "The model file best_model.pkl was not found. "
        "Place best_model.pkl in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load best_model.pkl: {e}")
    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================
st.markdown(
    """
    <div class="section-title">Soil Properties</div>
    <div class="section-description">
        Enter the values of the six input parameters.
    </div>
    """,
    unsafe_allow_html=True,
)

# Use dataset medians as initial values
defaults = {}

for feature in FEATURES:
    median_value = df[feature].median()
    defaults[feature] = (
        float(median_value) if pd.notna(median_value) else 0.0
    )

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        dry_density = st.number_input(
            "Dry density",
            value=defaults["Dry density"],
            format="%.6f",
            help="Enter the value using the scale of the training dataset.",
        )

        porosity = st.number_input(
            "Porosity",
            value=defaults["Porosity"],
            format="%.6f",
            help="Use the same porosity scale as the training dataset.",
        )

    with col2:
        saturation = st.number_input(
            "Saturation degree",
            value=defaults["Saturation degree"],
            format="%.6f",
            help="Use the same saturation scale as the training dataset.",
        )

        quartz = st.number_input(
            "Quartz content",
            value=defaults["Quartz content"],
            format="%.6f",
            help="Use the same quartz-content scale as the training dataset.",
        )

    with col3:
        sand = st.number_input(
            "Sand content",
            value=defaults["Sand content"],
            format="%.6f",
            help="Use the same sand-content scale as the training dataset.",
        )

        clay = st.number_input(
            "Clay content",
            value=defaults["Clay content"],
            format="%.6f",
            help="Use the same clay-content scale as the training dataset.",
        )

    submitted = st.form_submit_button(
        "Predict Thermal Conductivity",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ANALYTICAL EQUATION
# K = 2.740221 - 0.598635*Q + 3.078759*S
#     - 8.672656*n + 1.399861*Q^2
#     - 1.991327*S^2 + 6.718210*n^2
# ============================================================
def analytical_prediction(q, s, n):
    return (
        2.740221
        - 0.598635 * q
        + 3.078759 * s
        - 8.672656 * n
        + 1.399861 * (q ** 2)
        - 1.991327 * (s ** 2)
        + 6.718210 * (n ** 2)
    )


# ============================================================
# PREDICTION AND SHAP EXPLANATION
# ============================================================
if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Dry density": dry_density,
                "Porosity": porosity,
                "Saturation degree": saturation,
                "Quartz content": quartz,
                "Sand content": sand,
                "Clay content": clay,
            }
        ],
        columns=FEATURES,
    )

    try:
        # Machine learning prediction
        ml_prediction = float(np.asarray(model.predict(input_data)).ravel()[0])

        # Analytical equation prediction
        equation_prediction = float(
            analytical_prediction(
                q=quartz,
                s=saturation,
                n=porosity,
            )
        )

        st.markdown("---")

        st.markdown(
            """
            <div class="section-title">Prediction Results</div>
            <div class="section-description">
                Results obtained from the trained model and analytical equation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Machine Learning Prediction
                    </div>
                    <div class="result-value">
                        {ml_prediction:.6f}
                    </div>
                    <div class="result-note">
                        Gradient Boosting model
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Analytical Equation Prediction
                    </div>
                    <div class="result-value">
                        {equation_prediction:.6f}
                    </div>
                    <div class="result-note">
                        Proposed analytical equation
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show the difference between predictions
        difference = abs(ml_prediction - equation_prediction)

        st.write("")
        st.metric(
            label="Absolute difference between predictions",
            value=f"{difference:.6f}",
        )

        # ----------------------------------------------------
        # SHAP WATERFALL PLOT
        # ----------------------------------------------------
        st.markdown("---")

        st.markdown(
            """
            <div class="section-title">SHAP Explainability</div>
            <div class="section-description">
                The waterfall plot shows how each input feature moves
                the model output away from its expected baseline.
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            if not hasattr(model, "steps") or len(model.steps) < 2:
                raise ValueError(
                    "The loaded model does not expose the expected "
                    "preprocessing and estimator pipeline."
                )

            # PyCaret pipeline: preprocessing steps followed by estimator
            estimator = model.steps[-1][1]
            preprocessor = model[:-1]

            # Transform input using the model's preprocessing steps
            transformed_input = preprocessor.transform(input_data)

            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            transformed_input = np.asarray(transformed_input)

            if transformed_input.ndim == 1:
                transformed_input = transformed_input.reshape(1, -1)

            if transformed_input.shape[1] != len(FEATURES):
                raise ValueError(
                    "The number of transformed features does not match "
                    "the six expected input features."
                )

            # Use a small sample of the reference data as SHAP background
            background_data = df[FEATURES].sample(
                n=min(100, len(df)),
                random_state=123,
            )

            transformed_background = preprocessor.transform(background_data)

            if hasattr(transformed_background, "toarray"):
                transformed_background = transformed_background.toarray()

            transformed_background = np.asarray(transformed_background)

            # TreeExplainer for Gradient Boosting
            explainer = shap.TreeExplainer(
                estimator,
                data=transformed_background,
                feature_perturbation="interventional",
            )

            shap_values = explainer.shap_values(transformed_input)

            # Support common SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[0, :, 0]
            elif shap_values.ndim == 2:
                shap_values = shap_values[0]

            base_values = explainer.expected_value
            base_values = np.asarray(base_values).reshape(-1)
            base_value = float(base_values[0])

            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=transformed_input[0],
                feature_names=FEATURES,
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            shap.plots.waterfall(
                explanation,
                max_display=len(FEATURES),
                show=False,
            )

            fig = plt.gcf()
            fig.set_facecolor("white")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Positive SHAP values increase the model prediction; "
                "negative SHAP values decrease it. The explanation "
                "corresponds to the machine learning model, not the "
                "analytical equation."
            )

        except Exception as shap_error:
            st.warning(
                "The predictions were calculated, but the SHAP plot "
                f"could not be generated: {shap_error}"
            )

    except Exception as prediction_error:
        st.error(f"Prediction failed: {prediction_error}")


# ============================================================
# ANALYTICAL EQUATION DISPLAY
# ============================================================
st.markdown("---")

st.markdown(
    """
    <div class="section-title">Analytical Equation</div>
    <div class="section-description">
        The equation uses porosity, saturation degree, and quartz content.
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    r"""
    \begin{aligned}
    K ={}& 2.740221 - 0.598635Q + 3.078759S \\
         &- 8.672656n + 1.399861Q^2 \\
         &- 1.991327S^2 + 6.718210n^2
    \end{aligned}
    """
)

st.markdown(
    """
    Where:
    - \(K\) is the predicted thermal conductivity.
    - \(Q\) is quartz content.
    - \(S\) is saturation degree.
    - \(n\) is porosity.

    **Important:** Use the same variable scales used when the equation
    was developed. If the equation was fitted using proportions between
    0 and 1, do not enter percentages between 0 and 100.
    """
)


# ============================================================
# REFERENCE DATASET STATISTICS
# ============================================================
st.markdown("---")

st.markdown(
    """
    <div class="section-title">Reference Dataset Statistics</div>
    <div class="section-description">
        A brief statistical overview of the dataset used by this application.
    </div>
    """,
    unsafe_allow_html=True,
)

stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.metric("Number of valid records", f"{len(df):,}")

with stat_col2:
    st.metric("Number of input features", len(FEATURES))

with st.expander("View descriptive statistics", expanded=False):
    statistics = df[FEATURES].describe().T

    statistics = statistics.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std. Dev.",
            "min": "Minimum",
            "25%": "25th Percentile",
            "50%": "Median",
            "75%": "75th Percentile",
            "max": "Maximum",
        }
    )

    st.dataframe(
        statistics.round(4),
        use_container_width=True,
    )

with st.expander("View feature ranges", expanded=False):
    ranges = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Minimum": [df[col].min() for col in FEATURES],
            "Maximum": [df[col].max() for col in FEATURES],
            "Mean": [df[col].mean() for col in FEATURES],
        }
    )

    st.dataframe(
        ranges.round(6),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-title">
            Soil Thermal Conductivity Prediction &amp; Explainability
        </div>

        <div class="footer-affiliation">
            Faculty of Civil, Water and Environmental Engineering<br>
            Shahid Beheshti University<br>
            Dr. Mohammadnezhad's Research Team
        </div>

        <div class="footer-copyright">
            © Shahid Beheshti University — Research Application
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
