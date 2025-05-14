import os
import subprocess
from pdf2image import convert_from_path
import pytesseract

input_docx = "src_documents/ocr_sample.docx"
temp_pdf = input_docx.replace(".docx", "_temp.pdf")
ocr_pdf = "src_documents/ocr_sample_ocr.pdf"

# Conversion DOCX -> PDF avec libreoffice (compatible Linux)
if not os.path.exists(temp_pdf):
    try:
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", "--outdir",
            os.path.dirname(temp_pdf), input_docx
        ], check=True)
        # LibreOffice nomme le PDF avec le même nom de base que le DOCX
        generated_pdf = os.path.join(
            os.path.dirname(temp_pdf),
            os.path.basename(input_docx).replace(".docx", ".pdf")
        )
        os.rename(generated_pdf, temp_pdf)
    except Exception as e:
        raise ValueError(f"Échec de la conversion DOCX -> PDF : {input_docx}. Erreur : {e}")

# 2. Convertir le PDF en images (augmentation de la résolution)
images = convert_from_path(temp_pdf, dpi=400)

# 3. Générer un PDF OCRisé directement à partir des images
with open(ocr_pdf, "wb") as f:
    for img in images:
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang="fra+eng")
        f.write(pdf_bytes)

print(f"PDF OCRisé généré : {ocr_pdf}")

# Suppression du fichier temporaire PDF
if os.path.exists(temp_pdf):
    os.remove(temp_pdf)
    print(f"Fichier temporaire supprimé : {temp_pdf}")