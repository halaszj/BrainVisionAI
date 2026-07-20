# Streamlit Community Cloud Deployment

## Required Python version

Open the app in Streamlit Community Cloud and select:

**Manage app → Settings → Advanced settings → Python version → 3.11**

Then save and reboot the app.

The Python version for an existing Streamlit Community Cloud app is controlled
through the deployment settings. Repository runtime files may not change the
interpreter of an already-created deployment.

## Dependency files

- `requirements.txt` contains only the dependencies required by the currently
  deployed Streamlit application.
- `requirements-training.txt` contains the complete local/Colab training stack.

This separation prevents Streamlit Community Cloud from installing TensorFlow
before the trained model and inference engine are added.
