from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def create_pdf_report(
    original_image,
    gradcam_image,
    prediction,
    metadata,
    disclaimer,
):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setTitle("BrainVision AI Prediction Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 55, "BrainVision AI Prediction Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, height - 75, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    y = height - 110
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Prediction")
    y -= 20

    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, y, f"Predicted class: {prediction['predicted_class']}")
    y -= 17
    pdf.drawString(60, y, f"Confidence: {prediction['confidence']:.2%}")
    y -= 28

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Class probabilities")
    y -= 20

    pdf.setFont("Helvetica", 10)
    for label, probability in sorted(
        prediction["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        pdf.drawString(60, y, f"{label}: {probability:.2%}")
        y -= 15

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Images")
    y -= 15

    original_buffer = BytesIO()
    original_image.save(original_buffer, format="PNG")
    original_buffer.seek(0)

    gradcam_buffer = BytesIO()
    gradcam_image.save(gradcam_buffer, format="PNG")
    gradcam_buffer.seek(0)

    pdf.drawImage(
        ImageReader(original_buffer),
        50,
        y - 210,
        width=230,
        height=200,
        preserveAspectRatio=True,
        anchor="c",
    )
    pdf.drawImage(
        ImageReader(gradcam_buffer),
        320,
        y - 210,
        width=230,
        height=200,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(165, y - 225, "Original MRI")
    pdf.drawCentredString(435, y - 225, "Grad-CAM")

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, 95, "Disclaimer")
    pdf.setFont("Helvetica", 8)

    text = pdf.beginText(50, 80)
    text.setLeading(10)
    for line in [
        disclaimer,
        "The output should be reviewed only as part of an educational demonstration.",
    ]:
        text.textLine(line)
    pdf.drawText(text)

    if metadata:
        pdf.setFont("Helvetica", 7)
        pdf.drawRightString(
            width - 50,
            40,
            f"Model: {metadata.get('architecture', 'Unknown')} | "
            f"Version: {metadata.get('version', '5.0.0')}",
        )

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
