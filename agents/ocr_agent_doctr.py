from pdf2image import convert_from_path
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import numpy as np
import json
import os

# === Step 1: Convert PDF to images ===
def convert_pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi=dpi)

# === Step 2: Run DocTR OCR ===
def run_doctr_ocr(images):
    model = ocr_predictor(pretrained=True)
    np_images = [np.array(img) for img in images]
    return model(np_images)

# === Step 3: Structure OCR output ===
def extract_blocks_from_doctr_result(result):
    structured_data = []

    for page_idx, page in enumerate(result.pages):
        page_data = {
            "page_number": page_idx + 1,
            "blocks": []
        }

        for block_idx, block in enumerate(page.blocks):
            block_text = " ".join([w.value for l in block.lines for w in l.words])
            block_bbox = block.geometry
            page_data["blocks"].append({
                "id": f"block_{page_idx + 1}_{block_idx + 1}",
                "type": "paragraph",  # Later: add table/image classification
                "text": block_text,
                "bbox": block_bbox
            })

        structured_data.append(page_data)

    return structured_data

# === Step 4: Main Execution ===
def process_pdf(pdf_path, output_json="output.json"):
    images = convert_pdf_to_images(pdf_path)
    result = run_doctr_ocr(images)
    structured = extract_blocks_from_doctr_result(result)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"JSON saved to: {output_json}")

if __name__ == "__main__":
    path = "documents_a_traduire/SQ_13124846.pdf"
    process_pdf(path, "structured_output.json")