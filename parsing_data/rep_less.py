from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- read input (skip blank lines) ---
with open("report.txt", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# --- multi-pass collection (NEC -> POS -> FAIL -> SAMPLE -> PASS) ---
patterns = ["NEC", "POS", "FAIL", "SAMPLE", "PASS"]
buckets = {p: [] for p in patterns}
seen = set()  # avoid duplicates if a line contains multiple keywords

for p in patterns:
    for ln in lines:
        if ln in seen:
            continue
        if p in ln:              # simple match as requested; flip to .lower() if you want case-insensitive
            buckets[p].append(ln)
            seen.add(ln)

# --- build PDF ---
doc = SimpleDocTemplate(
    "sorted_report.pdf",
    pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=20*mm, bottomMargin=20*mm
)

styles = getSampleStyleSheet()
title_style = styles["Heading1"]
heading_style = styles["Heading2"]
body_style = ParagraphStyle(
    name="BodyMono",
    parent=styles["Normal"],
    fontName="Courier",   # monospace to preserve alignment
    fontSize=10,
    leading=12,
    alignment=TA_LEFT
)

story = []
story.append(Paragraph("Sorted Report", title_style))
story.append(Spacer(1, 10))

for p in patterns:
    if buckets[p]:
        story.append(Paragraph(p, heading_style))
        for ln in buckets[p]:
            story.append(Preformatted(ln, body_style))
        story.append(Spacer(1, 8))

doc.build(story)
print("PDF created: sorted_report.pdf")
