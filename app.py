from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "brainvision_efficientnetb0.keras"
LABELS_PATH = PROJECT_ROOT / "model" / "labels.json"

st.set_page_config(
    page_title="BrainVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bv-primary: #5b6cff;
        --bv-secondary: #8a5cff;
        --bv-surface: rgba(255,255,255,0.06);
        --bv-border: rgba(255,255,255,0.12);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .bv-hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid var(--bv-border);
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(138,92,255,0.28), transparent 34%),
            radial-gradient(circle at bottom left, rgba(91,108,255,0.24), transparent 38%),
            var(--bv-surface);
        box-shadow: 0 20px 55px rgba(0,0,0,0.20);
        margin-bottom: 1.5rem;
    }

    .bv-title {
        font-size: clamp(2.1rem, 5vw, 4rem);
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .bv-subtitle {
        margin-top: 0.8rem;
        font-size: 1.05rem;
        opacity: 0.84;
        max-width: 760px;
    }

    .bv-badge {
        display: inline-block;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        border: 1px solid var(--bv-border);
        background: rgba(91,108,255,0.15);
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .bv-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--bv-border);
        border-radius: 18px;
        background: var(--bv-surface);
        min-height: 150px;
    }

    .bv-card h3 {
        margin-top: 0;
        margin-bottom: 0.55rem;
        font-size: 1.08rem;
    }

    .bv-muted {
        opacity: 0.75;
        font-size: 0.94rem;
    }

    .bv-status-ready {
        color: #41d18b;
        font-weight: 700;
    }

    .bv-status-pending {
        color: #ffba49;
        font-weight: 700;
    }

    .bv-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--bv-border);
        opacity: 0.7;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_labels() -> dict:
    if not LABELS_PATH.exists():
        return {}
    try:
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

labels = load_labels()
model_available = MODEL_PATH.exists()

with st.sidebar:
    st.markdown("## BrainVision AI")
    st.caption("Graduate Deep Learning Portfolio Project")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "Model Status", "Project Architecture"],
        index=0,
    )

    st.divider()
    st.markdown("### Classification Classes")
    st.markdown(
        """
        - Glioma
        - Meningioma
        - Pituitary
        - No Tumor
        """
    )

    st.divider()
    st.caption("Educational and research use only. Not for clinical diagnosis.")

if page == "Home":
    st.markdown(
        """
        <section class="bv-hero">
            <div class="bv-badge">TensorFlow • EfficientNetB0 • Grad-CAM</div>
            <h1 class="bv-title">BrainVision AI</h1>
            <p class="bv-subtitle">
                A professional brain MRI classification platform designed to distinguish
                glioma, meningioma, pituitary tumors, and scans showing no tumor.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(3)

    with left:
        st.markdown(
            """
            <div class="bv-card">
                <h3>EfficientNetB0</h3>
                <div class="bv-muted">
                    ImageNet transfer learning with staged fine-tuning for efficient,
                    high-quality MRI feature extraction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.markdown(
            """
            <div class="bv-card">
                <h3>Explainable Predictions</h3>
                <div class="bv-muted">
                    Grad-CAM visualizations will highlight image regions that contributed
                    most strongly to each prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="bv-card">
                <h3>Portfolio-Ready Reporting</h3>
                <div class="bv-muted">
                    Interactive Plotly analytics and downloadable PDF prediction reports
                    are included in the full application build.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Deployment Status")

    if model_available:
        st.success(
            "The trained model is available. The inference interface will be enabled "
            "when the remaining application modules are added."
        )
    else:
        st.warning(
            "The Streamlit application is running correctly, but the trained model has "
            "not been added yet. Run `training/01_train_efficientnet.ipynb`, then place "
            "`brainvision_efficientnetb0.keras` in the repository's `model/` directory."
        )

    st.info(
        "Build 1 establishes the validated training pipeline and deployment shell. "
        "The next build adds production inference, preprocessing, and Grad-CAM."
    )

elif page == "Model Status":
    st.title("Model Status")

    status_col, path_col = st.columns([1, 2])

    with status_col:
        if model_available:
            st.metric("Model", "Available")
            st.markdown(
                '<span class="bv-status-ready">Ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Model", "Not Found")
            st.markdown(
                '<span class="bv-status-pending">Training required</span>',
                unsafe_allow_html=True,
            )

    with path_col:
        st.markdown("#### Expected model file")
        st.code(str(MODEL_PATH.relative_to(PROJECT_ROOT)))

        st.markdown("#### Expected labels file")
        st.code(str(LABELS_PATH.relative_to(PROJECT_ROOT)))

    st.markdown("### Loaded Label Metadata")
    if labels:
        st.json(labels)
    else:
        st.caption("No exported labels file is present yet.")

elif page == "Project Architecture":
    st.title("Project Architecture")

    st.markdown(
        """
        ```text
        MRI image
            ↓
        Validation and preprocessing
            ↓
        EfficientNetB0 feature extractor
            ↓
        Classification head
            ↓
        Four-class probability output
            ↓
        Grad-CAM explanation
            ↓
        Plotly dashboard and PDF report
        ```
        """
    )

    st.markdown("### Current Repository Components")
    st.markdown(
        """
        - Complete EfficientNetB0 training notebook
        - Centralized YAML training configuration
        - Model and artifact export paths
        - Streamlit deployment entry point
        - Python 3.11 runtime configuration
        """
    )

st.markdown(
    """
    <div class="bv-footer">
        BrainVision AI • Graduate Deep Learning Project • Educational use only
    </div>
    """,
    unsafe_allow_html=True,
)
