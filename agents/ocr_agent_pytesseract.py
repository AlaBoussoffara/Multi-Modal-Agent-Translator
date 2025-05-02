import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import os

# Optional: set path to Tesseract executable if needed (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi=dpi)

def extract_blocks_from_image(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    blocks = {}
    
    n = len(data['text'])
    for i in range(n):
        block_num = data['block_num'][i]
        text = data['text'][i].strip()
        
        if text:
            bbox = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            if block_num not in blocks:
                blocks[block_num] = {
                    'block_num': block_num,
                    'bbox': bbox,
                    'text': text
                }
            else:
                blocks[block_num]['text'] += ' ' + text
                # Optionally, expand bbox if needed

    return list(blocks.values())

def process_pdf(pdf_path, output_json="output_pytesseract.json"):
    images = convert_pdf_to_images(pdf_path)
    structured_data = []

    for page_idx, image in enumerate(images):
        page_data = {
            "page_number": page_idx + 1,
            "blocks": extract_blocks_from_image(image)
        }
        structured_data.append(page_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"OCR results saved to: {output_json}")

if __name__ == "__main__":
    path = "documents_a_traduire/SQ_13124846.pdf"
    process_pdf(path, "structured_output_pytesseract.json")
