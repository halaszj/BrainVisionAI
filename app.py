import json
from datetime import datetime
from pathlib import Path

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
from brainvision.inference import (
    load_labels,
    load_metadata,
    predict_image,
)
from brainvision.gradcam import make_gradcam_heatmap, overlay_heatmap
from brainvision.reporting import create_pdf_report

st.set_page_config(
    page_title="BrainVision AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0;
    }
    .version-line {
        color: #666;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_application_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Copy the exported model from Delivery 2 into the model folder."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found at {LABELS_PATH}."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    metadata = load_metadata(METADATA_PATH)
    return model, labels, metadata

def initialize_session():
    if "history" not in st.session_state:
        st.session_state.history = []

initialize_session()

st.markdown('<div class="main-title">🧠 BrainVision AI</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="version-line">Version {VERSION} • Build {BUILD} • Delivery {DELIVERY}</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["MRI Analysis", "Session History", "Model Information", "About"],
    )
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
    st.subheader("Upload a Brain MRI Image")
    st.write(
        "Upload one MRI image to generate a four-class prediction, "
        "confidence scores, Grad-CAM visualization, and a PDF report."
    )

    if not model_ready:
        st.error(load_error)
        st.info(
            "Expected files:\n\n"
            "- `model/brainvision_efficientnetb0_final.keras`\n"
            "- `model/labels.json`\n"
            "- `model/model_metadata.json` (optional)"
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "MRI image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Use a clear MRI image in PNG, JPG, JPEG, or BMP format.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
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

            left, right = st.columns(2)

            with left:
                st.image(
                    prediction["display_image"],
                    caption="Uploaded MRI",
                    use_container_width=True,
                )

            with right:
                st.image(
                    gradcam_image,
                    caption=f"Grad-CAM • Layer: {layer_name}",
                    use_container_width=True,
                )

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h3>{prediction["predicted_class"].title()}</h3>
                    <p><strong>Confidence:</strong> {prediction["confidence"]:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            probability_df = pd.DataFrame(
                {
                    "Class": list(prediction["probabilities"].keys()),
                    "Probability": list(prediction["probabilities"].values()),
                }
            ).sort_values("Probability", ascending=False)

            st.subheader("Class Probabilities")
            st.bar_chart(
                probability_df.set_index("Class"),
                use_container_width=True,
            )
            st.dataframe(
                probability_df.assign(
                    Probability=probability_df["Probability"].map(
                        lambda value: f"{value:.2%}"
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "prediction": prediction["predicted_class"],
                "confidence": prediction["confidence"],
            }

            if not st.session_state.history or (
                st.session_state.history[-1] != record
            ):
                st.session_state.history.append(record)

            pdf_bytes = create_pdf_report(
                original_image=prediction["display_image"],
                gradcam_image=gradcam_image,
                prediction=prediction,
                metadata=metadata,
                disclaimer=MEDICAL_DISCLAIMER,
            )

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="BrainVisionAI_Prediction_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.warning(MEDICAL_DISCLAIMER)

        except Exception as error:
            st.exception(error)

elif page == "Session History":
    st.subheader("Session Prediction History")

    if not st.session_state.history:
        st.info("No predictions have been made during this session.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["confidence"] = history_df["confidence"].map(
            lambda value: f"{value:.2%}"
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Session History",
            data=history_csv,
            file_name="BrainVisionAI_Session_History.csv",
            mime="text/csv",
        )

        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()

elif page == "Model Information":
    st.subheader("Model Information")

    if not model_ready:
        st.error(load_error)
    else:
        info = {
            "Project": metadata.get("project_name", "BrainVision AI"),
            "Version": metadata.get("version", VERSION),
            "Build": metadata.get("build", BUILD),
            "Architecture": metadata.get("architecture", "EfficientNetB0"),
            "Image size": str(metadata.get("image_size", list(IMAGE_SIZE))),
            "Classes": ", ".join(metadata.get("class_names", labels)),
            "Test accuracy": (
                f"{metadata['test_accuracy']:.2%}"
                if "test_accuracy" in metadata
                else "Not provided"
            ),
            "Created": metadata.get("created_at", "Not provided"),
        }

        st.dataframe(
            pd.DataFrame(
                {"Property": list(info.keys()), "Value": list(info.values())}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Model Summary")
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        st.code("\n".join(summary_lines), language="text")

elif page == "About":
    st.subheader("About BrainVision AI")
    st.write(
        "BrainVision AI is an educational deep-learning project that "
        "classifies brain MRI images into four categories using an "
        "EfficientNetB0 transfer-learning model."
    )

    st.markdown(
        """
        **Main features**

        - Brain MRI upload
        - Four-class prediction
        - Confidence visualization
        - Grad-CAM explanation
        - Downloadable PDF report
        - Session history
        - Model metadata display
        """
    )

    st.warning(MEDICAL_DISCLAIMER)

st.divider()
st.caption(f"BrainVision AI • Version {VERSION} • Build {BUILD} • Delivery {DELIVERY}")
