# Delivery 4 Guide

## 1. Get the model from Delivery 2

Run the Delivery 2 notebook completely. Download the output ZIP and extract it.

## 2. Copy the model

Copy these files into the Delivery 4 `model` folder:

- `brainvision_efficientnetb0_final.keras`
- `labels.json`
- `model_metadata.json`

## 3. Run the application

Install the dependencies and run:

```bash
streamlit run app.py
```

## 4. Test it

Upload one MRI image from the dataset's Testing folder. The application should show:

- the original image
- the predicted class
- the confidence value
- class probabilities
- the Grad-CAM overlay
- a PDF-download button
