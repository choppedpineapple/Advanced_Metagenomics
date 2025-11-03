from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# Read the file
with open("report.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # skip empty lines

# Sort by priority
priority = ["NEC", "POS", "FAIL", "SAMPLE", "PASS"]
buckets = {key: [] for key in priority}

for line in lines:
    for key in priority:
        if key in line:
            buckets[key].append(line)
            break

# Create the PDF
doc = SimpleDocTemplate(
    "sorted_report.pdf",
    pagesize=A4,
    leftMargin=20 * mm,
    rightMargin=20 * mm,
    topMargin=20 * mm,
    bottomMargin=20 * mm,
)

styles = getSampleStyleSheet()
title_style = styles["Heading1"]
heading_style = styles["Heading2"]
body_style = ParagraphStyle(
    name="Body",
    parent=styles["Normal"],
    fontName="Courier",
    fontSize=10,
    leading=12,
    alignment=TA_LEFT,
)

story = []
story.append(Paragraph("Sorted Report", title_style))
story.append(Spacer(1, 10))

# Write each section
for key in priority:
    if buckets[key]:
        story.append(Paragraph(key, heading_style))
        for line in buckets[key]:
            story.append(Preformatted(line, body_style))
        story.append(Spacer(1, 8))

doc.build(story)
print("✅ PDF created successfully: sorted_report.pdf")
