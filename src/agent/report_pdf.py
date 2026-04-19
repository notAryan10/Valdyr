from fpdf import FPDF
import re

def clean_text(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text

def generate_pdf(report_text, predicted_price, property_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Valdyr - Real Estate Advisory Report", align="C")
    pdf.ln(15)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Predicted Price: Rs {int(predicted_price):,}", align="C")
    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Property Details")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)

    labels = {
        "area": "Area (sq ft)", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
        "stories": "Stories", "parking": "Parking", "mainroad": "Main Road",
        "guestroom": "Guest Room", "basement": "Basement",
        "hotwaterheating": "Hot Water", "airconditioning": "AC",
        "prefarea": "Preferred Area", "furnishingstatus": "Furnishing"
    }

    for key, label in labels.items():
        if key in property_data:
            val = property_data[key]
            if key == "furnishingstatus":
                display = str(val)
            elif val in [0, 1] and key != "area":
                display = "Yes" if val else "No"
            else:
                display = str(val)
            pdf.cell(0, 6, f"  {label}: {display}")
            pdf.ln(6)

    pdf.ln(5)

    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        heading = re.match(r'^#{1,3}\s+(.+)$', line)
        numbered = re.match(r'^\d+\.\s+\*\*(.+?)\*\*', line)

        if heading:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, clean_text(heading.group(1)))
            pdf.ln(10)
            pdf.set_font("Helvetica", "", 10)
        elif numbered:
            pdf.set_font("Helvetica", "B", 10)
            pdf.x = pdf.l_margin
            pdf.multi_cell(0, 6, clean_text(line))
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.x = pdf.l_margin
            pdf.multi_cell(0, 6, clean_text(line))

    return bytes(pdf.output())
