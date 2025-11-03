from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# --- read input (skip blank lines) ---
with open("report.txt", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# normalize helper
def U(s): return s.upper()

# Buckets
NEC, POS, FAIL_SAMPLES, PASS_SAMPLES = [], [], [], []
seen = set()

# 1) Controls first
for ln in lines:
    u = U(ln)
    if "NEC" in u:
        NEC.append(ln); seen.add(ln)

for ln in lines:
    if ln in seen: continue
    u = U(ln)
    if "POS" in u:
        POS.append(ln); seen.add(ln)

# 2) Sample FAIL (must contain SAMPLE + FAIL; exclude controls)
for ln in lines:
    if ln in seen: continue
    u = U(ln)
    if ("SAMPLE" in u) and ("FAIL" in u) and ("NEC" not in u) and ("POS" not in u):
        FAIL_SAMPLES.append(ln); seen.add(ln)

# 3) Sample PASS (must contain SAMPLE + PASS; exclude controls)
for ln in lines:
    if ln in seen: continue
    u = U(ln)
    if ("SAMPLE" in u) and ("PASS" in u) and ("NEC" not in u) and ("POS" not in u):
        PASS_SAMPLES.append(ln); seen.add(ln)

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
    fontName="Courier",
    fontSize=10,
    leading=12,
    alignment=TA_LEFT
)

story = []
story.append(Paragraph("Sorted Report", title_style))
story.append(Spacer(1, 10))

sections = [
    ("NEC (Negative Control)", NEC),
    ("POS (Positive Control)", POS),
    ("FAIL (Samples)", FAIL_SAMPLES),
    ("PASS (Samples)", PASS_SAMPLES),
]

for title, items in sections:
    if items:
        story.append(Paragraph(title, heading_style))
        for ln in items:
            story.append(Preformatted(ln, body_style))
        story.append(Spacer(1, 8))

doc.build(story)
print(" PDF created successfully: sorted_report.pdf")
