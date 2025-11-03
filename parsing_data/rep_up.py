from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- Read lines ---
with open("report.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # skip empty lines

# --- Sort lines based on priority ---
priority = ["NEC", "POS", "FAIL", "SAMPLE", "PASS"]
buckets = {key: [] for key in priority}

for line in lines:
    placed = False
    for key in priority:
        if key in line:
            buckets[key].append(line)
            placed = True
            break
    if not placed:
        # if it doesn't match any known keyword, put it under 'OTHER'
        buckets.setdefault("OTHER", []).append(line)

# --- Create PDF ---
doc = SimpleDocTemplate("sorted_report.pdf", pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=20 * mm, bottomMargin=20 * mm)

styles = getSampleStyleSheet()
h2 = styles["Heading2"]
body_style = ParagraphStyle(
    name="Body",
    parent=styles["Normal"],
    fontName="Courier",
    fontSize=10,
    leading=12,
    alignment=TA_LEFT
)

story = []
story.append(Paragraph("Sorted Report", styles["Heading1"]))
story.append(Spacer(1, 10))

for key in priority + (["OTHER"] if "OTHER" in buckets else []):
    if buckets[key]:
        story.append(Paragraph(key, h2))
        for line in buckets[key]:
            story.append(Preformatted(line, body_style))
        story.append(Spacer(1, 8))

doc.build(story)
print("PDF created successfully: sorted_report.pdf")
