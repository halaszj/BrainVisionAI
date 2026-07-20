# Build 3 Deployment Instructions

Build 3 has two deployment states.

## State 1: Interface deployment

The root `requirements.txt` installs the complete upload interface, Plotly
analytics, and PDF reporting layer without TensorFlow. This lets the application
remain online while the model is being trained.

The application will honestly report:

- Model artifact not installed
- TensorFlow runtime not installed
- Prediction setup required

It will not produce fake predictions.

## State 2: Full inference deployment

Complete these steps after training:

1. Run `training/01_train_efficientnet.ipynb`.
2. Run `training/03_export_model.ipynb`.
3. Add the generated file to the repository:

```text
model/brainvision_efficientnetb0_deployment.keras
```

4. Delete and redeploy the Streamlit Community Cloud application using Python
   3.11 if the existing deployment uses another Python version.
5. Replace the contents of `requirements.txt` with the contents of
   `requirements-inference.txt`.
6. Commit and push the dependency change.
7. Wait for Streamlit Community Cloud to rebuild.

The application will automatically detect the model and enable the Analyze MRI
button when both the model artifact and TensorFlow runtime are available.

## Why the dependencies are separated

TensorFlow is large and unnecessary until a trained model is present. Keeping it
out of the interface-only deployment prevents needless installation failures and
keeps the working project page available.
