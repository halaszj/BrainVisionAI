# BrainVision AI

BrainVision AI is a graduate-level deep learning project that classifies brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project uses TensorFlow/Keras, EfficientNetB0 transfer learning, Grad-CAM explainability, Plotly dashboards, PDF report generation, and a professional Streamlit interface.

> **Important:** BrainVision AI is an educational research application. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Build 1 Contents

This package contains the complete model-training foundation:

```text
BrainVisionAI_Build1/
├── training/
│   └── 01_train_efficientnet.ipynb
├── config/
│   └── training_config.yaml
├── model/
│   └── .gitkeep
├── artifacts/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## Dataset

Use the **Masoud Nickparvar Brain Tumor MRI Dataset**. The notebook supports the standard Kaggle directory structure:

```text
data/brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The notebook also searches several common local and Google Colab locations automatically. You may set `DATASET_ROOT` in the notebook when your dataset is stored elsewhere.

## Environment Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Training Notebook

From the repository root:

```bash
jupyter lab training/01_train_efficientnet.ipynb
```

Run every cell in order. The notebook performs:

1. Environment validation and reproducibility setup
2. Dataset discovery and integrity checks
3. Class-distribution analysis
4. TensorFlow input-pipeline construction
5. Data augmentation
6. EfficientNetB0 transfer learning
7. Fine-tuning of upper backbone layers
8. Training diagnostics
9. Test-set evaluation
10. Confusion matrix and classification report
11. One-vs-rest ROC and precision-recall curves
12. Misclassification analysis
13. Model and metadata export

## Training Strategy

The model is trained in two stages:

### Stage 1: Classification Head

The ImageNet-pretrained EfficientNetB0 backbone is frozen. A custom classification head is trained using:

- Global average pooling
- Batch normalization
- Dense layer with Swish activation
- Dropout
- Four-class softmax output

### Stage 2: Fine-Tuning

Upper EfficientNetB0 layers are unfrozen while Batch Normalization layers remain frozen. Fine-tuning uses a lower learning rate to preserve useful pretrained features.

## Generated Outputs

After successful execution, the notebook writes:

```text
model/
├── brainvision_efficientnetb0.keras
├── brainvision_efficientnetb0_final.keras
└── labels.json

artifacts/
├── training_history.csv
├── test_metrics.json
├── classification_report.csv
├── confusion_matrix.csv
├── predictions.csv
├── run_metadata.json
├── training_curves.png
├── confusion_matrix.png
├── roc_curves.png
├── precision_recall_curves.png
└── misclassified_examples.png
```

## Reproducibility

The notebook sets deterministic random seeds for Python, NumPy, and TensorFlow. Exact numerical results can still vary slightly across hardware, TensorFlow versions, GPU kernels, and operating systems.

## Streamlit Community Cloud

The final application will be deployed through Streamlit Community Cloud. The repository uses `runtime.txt` with Python 3.11 for TensorFlow compatibility. When deploying, confirm Python 3.11 in the Streamlit advanced settings.

## Repository Roadmap

- [x] Build 1: EfficientNetB0 training notebook and base configuration
- [ ] Build 2: Grad-CAM notebook, model export, and inference utilities
- [ ] Build 3: Streamlit application, dashboards, PDF reports, tests, and deployment documentation

## License

A project license will be included in the final repository package.
