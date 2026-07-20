from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _pil_to_png_bytes(image: Image.Image) -> BytesIO:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def build_prediction_pdf(
    *,
    prediction: dict[str, Any],
    original_image: Image.Image,
    overlay_image: Image.Image | None = None,
) -> bytes:
    """Create an educational BrainVision AI prediction report."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="BrainVision AI Prediction Report",
        author="BrainVision AI",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BVTitle",
            parent=styles["Title"],
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#26304A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BVSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5D6474"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BVHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#38466A"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BVWarning",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#7A3F00"),
            backColor=colors.HexColor("#FFF4DE"),
            borderColor=colors.HexColor("#F3C77C"),
            borderWidth=0.6,
            borderPadding=8,
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    story = [
        Paragraph("BrainVision AI", styles["BVTitle"]),
        Paragraph(
            "Brain MRI Classification — Educational Prediction Report",
            styles["BVSubtitle"],
        ),
    ]

    summary_data = [
        ["Predicted class", prediction["predicted_display_name"]],
        ["Confidence", f'{prediction["confidence"]:.2%}'],
        ["Model version", prediction["model_version"]],
        ["Generated", prediction["timestamp"]],
        ["Source file", prediction.get("source_filename") or "Not provided"],
        [
            "Image dimensions",
            f'{prediction["image_width"]} × {prediction["image_height"]} pixels',
        ],
    ]
    summary = Table(summary_data, colWidths=[1.65 * inch, 4.85 * inch])
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF1FA")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#38466A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD3E3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary, Spacer(1, 12)])

    original_buffer = _pil_to_png_bytes(original_image)
    original_report_image = ReportImage(original_buffer, width=2.75 * inch, height=2.75 * inch)

    if overlay_image is not None:
        overlay_buffer = _pil_to_png_bytes(overlay_image)
        overlay_report_image = ReportImage(
            overlay_buffer,
            width=2.75 * inch,
            height=2.75 * inch,
        )
        images = Table(
            [
                [
                    Paragraph("<b>Original MRI</b>", styles["Normal"]),
                    Paragraph("<b>Grad-CAM Overlay</b>", styles["Normal"]),
                ],
                [original_report_image, overlay_report_image],
            ],
            colWidths=[3.2 * inch, 3.2 * inch],
        )
    else:
        images = Table(
            [
                [Paragraph("<b>Original MRI</b>", styles["Normal"])],
                [original_report_image],
            ],
            colWidths=[3.2 * inch],
        )

    images.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 1), (-1, -1), 0.5, colors.HexColor("#CCD3E3")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(KeepTogether(images))

    story.append(Paragraph("Class Probabilities", styles["BVHeading"]))
    probability_rows = [["Class", "Probability"]]
    for item in prediction["probabilities"]:
        probability_rows.append(
            [item["display_name"], f'{item["probability"]:.2%}']
        )

    probability_table = Table(probability_rows, colWidths=[4.8 * inch, 1.7 * inch])
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#38466A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD3E3")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(probability_table)

    story.append(
        Paragraph(
            "<b>Important:</b> This output is produced by a student deep-learning "
            "project for educational and research purposes. It is not a medical "
            "diagnosis, has not been clinically validated, and must not be used to "
            "make treatment decisions. A qualified medical professional should "
            "interpret all MRI findings.",
            styles["BVWarning"],
        )
    )

    document.build(story)
    output.seek(0)
    return output.getvalue()
