from datetime import datetime

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from brainvision.version import VERSION, BUILD, DELIVERY
from brainvision.config import (
    MODEL_PATH,
    LABELS_PATH,
    METADATA_PATH,
    IMAGE_SIZE,
    MEDICAL_DISCLAIMER,
)
from brainvision.inference import load_labels, load_metadata, predict_image
from brainvision.gradcam import make_gradcam_heatmap, overlay_heatmap
from brainvision.reporting import create_pdf_report

st.set_page_config(
    page_title="BrainVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 5%, rgba(124,77,255,.10), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(0,188,212,.08), transparent 25%),
            linear-gradient(180deg, #f8f9fd 0%, #ffffff 50%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17152b 0%, #252047 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f7f5ff;
    }

    .hero {
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
        background:
            linear-gradient(120deg, rgba(42,31,89,.96), rgba(91,58,170,.93)),
            linear-gradient(45deg, #24184c, #6045ba);
        box-shadow: 0 18px 45px rgba(52, 36, 105, .20);
        color: white;
        overflow: hidden;
        position: relative;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        right: -70px;
        top: -100px;
        background: rgba(255,255,255,.09);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.65rem;
        line-height: 1.05;
        color: white;
    }

    .hero p {
        max-width: 760px;
        margin: .8rem 0 0;
        color: rgba(255,255,255,.84);
        font-size: 1.02rem;
    }

    .version-pill {
        display: inline-block;
        margin-top: 1rem;
        padding: .38rem .75rem;
        border-radius: 999px;
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.18);
        font-size: .82rem;
        color: white;
    }

    .glass-card {
        border: 1px solid rgba(98,76,177,.13);
        background: rgba(255,255,255,.88);
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 12px 35px rgba(60,49,105,.09);
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 20px;
        padding: 1.3rem 1.45rem;
        color: white;
        background: linear-gradient(135deg, #5f3dc4, #8b5cf6);
        box-shadow: 0 15px 35px rgba(93,61,196,.24);
        margin: .5rem 0 1.2rem;
    }

    .result-card .label {
        opacity: .78;
        font-size: .78rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .result-card .prediction {
        font-size: 1.85rem;
        font-weight: 750;
        margin-top: .2rem;
    }

    .result-card .confidence {
        font-size: 1rem;
        margin-top: .35rem;
        opacity: .92;
    }

    .mini-card {
        border-radius: 16px;
        padding: 1rem;
        background: #ffffff;
        border: 1px solid rgba(98,76,177,.12);
        box-shadow: 0 8px 22px rgba(60,49,105,.07);
        min-height: 112px;
    }

    .mini-card .eyebrow {
        color: #756c96;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    .mini-card .value {
        color: #2d2649;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: .35rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 720;
        color: #2f2850;
        margin: .6rem 0 .8rem;
    }

    .soft-note {
        border-radius: 14px;
        background: #f2efff;
        border: 1px solid #ded6ff;
        color: #51467d;
        padding: .85rem 1rem;
        margin: .7rem 0 1rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 18px;
        padding: .5rem;
        background: rgba(255,255,255,.72);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(98,76,177,.11);
        border-radius: 16px;
        padding: .85rem 1rem;
        box-shadow: 0 8px 22px rgba(60,49,105,.06);
    }

    .footer {
        text-align: center;
        color: #81799a;
        font-size: .82rem;
        padding: 1.5rem 0 .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_application_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Copy the trained model from Delivery 2."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}.")

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    metadata = load_metadata(METADATA_PATH)
    return model, labels, metadata


def initialize_session():
    if "history" not in st.session_state:
        st.session_state.history = []


def hero():
    st.markdown(
        f"""
        <div class="hero">
            <h1>🧠 BrainVision AI</h1>
            <p>
                Upload a brain MRI image to explore the model's predicted class,
                confidence levels, and Grad-CAM explanation.
            </p>
            <span class="version-pill">
                Version {VERSION} &nbsp;•&nbsp; Build {BUILD} &nbsp;•&nbsp; Delivery 4.1
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


initialize_session()
hero()

with st.sidebar:
    st.markdown("## BrainVision AI")
    st.caption("Deep-learning MRI classifier")
    page = st.radio(
        "Navigation",
        ["MRI Analysis", "Session History", "Model Information", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Model classes")
    st.caption("Glioma · Meningioma · No Tumor · Pituitary")
    st.divider()
    st.caption(MEDICAL_DISCLAIMER)

try:
    model, labels, metadata = load_application_model()
    model_ready = True
except Exception as error:
    model_ready = False
    model = labels = metadata = None
    load_error = str(error)

if page == "MRI Analysis":
    st.markdown('<div class="section-title">MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-note">
            Choose a clear brain MRI image. The application will show the model's
            prediction, confidence distribution, and a Grad-CAM heatmap.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(load_error)
        st.info(
            "Place `brainvision_efficientnetb0_final.keras` and `labels.json` "
            "inside the `model` folder."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="PNG, JPG, JPEG, or BMP",
    )

    if uploaded_file is None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="mini-card"><div class="eyebrow">Step 1</div>'
                '<div class="value">Upload MRI</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="mini-card"><div class="eyebrow">Step 2</div>'
                '<div class="value">Review Prediction</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="mini-card"><div class="eyebrow">Step 3</div>'
                '<div class="value">Download Report</div></div>',
                unsafe_allow_html=True,
            )
    else:
        try:
            original_image = Image.open(uploaded_file)
            prediction = predict_image(
                model=model,
                labels=labels,
                image=original_image,
                image_size=IMAGE_SIZE,
            )

            heatmap, layer_name = make_gradcam_heatmap(
                model=model,
                processed_batch=prediction["batch"],
                class_index=prediction["predicted_index"],
            )
            gradcam_image = overlay_heatmap(
                prediction["display_image"],
                heatmap,
                alpha=0.40,
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Predicted class</div>
                    <div class="prediction">{prediction["predicted_class"].title()}</div>
                    <div class="confidence">
                        Confidence: {prediction["confidence"]:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            image_col, gradcam_col = st.columns(2, gap="large")

            with image_col:
                st.markdown("#### Original MRI")
                st.image(
                    prediction["display_image"],
                    use_container_width=True,
                )

            with gradcam_col:
                st.markdown("#### Grad-CAM Explanation")
                st.image(
                    gradcam_image,
                    use_container_width=True,
                )
                st.caption(f"Feature layer: {layer_name}")

            probability_df = pd.DataFrame(
                {
                    "Class": list(prediction["probabilities"].keys()),
                    "Probability": list(prediction["probabilities"].values()),
                }
            ).sort_values("Probability", ascending=False)

            st.markdown(
                '<div class="section-title">Confidence Breakdown</div>',
                unsafe_allow_html=True,
            )

            metric_columns = st.columns(len(probability_df))
            for column, row in zip(metric_columns, probability_df.itertuples()):
                with column:
                    st.metric(row.Class.title(), f"{row.Probability:.1%}")

            st.bar_chart(
                probability_df.set_index("Class"),
                use_container_width=True,
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "prediction": prediction["predicted_class"],
                "confidence": prediction["confidence"],
            }

            if not st.session_state.history or st.session_state.history[-1] != record:
                st.session_state.history.append(record)

            pdf_bytes = create_pdf_report(
                original_image=prediction["display_image"],
                gradcam_image=gradcam_image,
                prediction=prediction,
                metadata=metadata,
                disclaimer=MEDICAL_DISCLAIMER,
            )

            st.download_button(
                "⬇ Download Prediction Report",
                data=pdf_bytes,
                file_name="BrainVisionAI_Prediction_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

            st.warning(MEDICAL_DISCLAIMER)

        except Exception as error:
            st.exception(error)

elif page == "Session History":
    st.markdown(
        '<div class="section-title">Session History</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No MRI images have been analyzed during this session.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        display_df = history_df.copy()
        display_df["confidence"] = display_df["confidence"].map(
            lambda value: f"{value:.2%}"
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download History CSV",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name="BrainVisionAI_Session_History.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with right:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

elif page == "Model Information":
    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(load_error)
    else:
        info = {
            "Project": metadata.get("project_name", "BrainVision AI"),
            "Version": metadata.get("version", VERSION),
            "Architecture": metadata.get("architecture", "EfficientNetB0"),
            "Image size": str(metadata.get("image_size", list(IMAGE_SIZE))),
            "Classes": ", ".join(metadata.get("class_names", labels)),
            "Test accuracy": (
                f"{metadata['test_accuracy']:.2%}"
                if "test_accuracy" in metadata else "Not provided"
            ),
            "Created": metadata.get("created_at", "Not provided"),
        }

        cols = st.columns(3)
        entries = list(info.items())
        for index, (label, value) in enumerate(entries):
            with cols[index % 3]:
                st.markdown(
                    f'<div class="mini-card"><div class="eyebrow">{label}</div>'
                    f'<div class="value">{value}</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")

        with st.expander("View full Keras model summary"):
            summary_lines = []
            model.summary(print_fn=summary_lines.append)
            st.code("\n".join(summary_lines), language="text")

elif page == "About":
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <h3>What BrainVision AI Does</h3>
                <p>
                    BrainVision AI is an educational image-classification project
                    built with TensorFlow, EfficientNetB0, and Streamlit.
                </p>
                <p>
                    It classifies uploaded MRI images into glioma, meningioma,
                    pituitary tumor, or no-tumor categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Included Features</h3>
                <p>✓ MRI upload</p>
                <p>✓ Confidence scores</p>
                <p>✓ Grad-CAM heatmap</p>
                <p>✓ PDF report</p>
                <p>✓ Session history</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(MEDICAL_DISCLAIMER)

st.markdown(
    f'<div class="footer">BrainVision AI • Version {VERSION} • Build {BUILD}</div>',
    unsafe_allow_html=True,
)
