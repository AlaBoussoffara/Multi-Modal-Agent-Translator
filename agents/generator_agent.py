# generator_agent.py

from abc import ABC, abstractmethod
import fitz
from collections import Counter
import numpy as np

class BaseOutputGenerator(ABC):
    """
    Abstract base class for generating output files with translated content.
    """
    @abstractmethod
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Reinserts translated content into the original layout and saves the output file.
        """
        pass

class PDFGenerator(BaseOutputGenerator):
    """
    Handles PDF output generation using PyMuPDF, including background sampling,
    redaction, and insertion of translated text.
    """

    def get_background_color_from_bbox(self, pdf_path, page_number, bbox,
                                       zoom=2, border_width=5, exclude_color=None,
                                       tolerance=20):
        """
        Determines the background color of a given bounding box by sampling border pixels.
        Excludes pixels close to the provided exclude_color (RGB) within the tolerance.
        """
        # Render the page at increased resolution
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert bbox from PDF points to pixel coordinates
        x0, y0, x1, y1 = [int(coord * zoom) for coord in bbox]
        cropped = img.crop((x0, y0, x1, y1))
        arr = np.array(cropped)  # shape: (height, width, 3)
        height, width, _ = arr.shape

        # Sample border pixels (top, bottom, left, right)
        border_pixels = []
        # Top
        border_pixels.append(arr[0:border_width, :, :].reshape(-1, 3))
        # Bottom
        border_pixels.append(arr[-border_width:, :, :].reshape(-1, 3))
        if height > 2 * border_width:
            # Left
            border_pixels.append(arr[border_width:height-border_width, 0:border_width, :].reshape(-1, 3))
            # Right
            border_pixels.append(arr[border_width:height-border_width, -border_width:, :].reshape(-1, 3))
        border_pixels = np.concatenate(border_pixels, axis=0)
        pixel_tuples = [tuple(pixel) for pixel in border_pixels]

        # Optionally exclude pixels near the provided font color
        if exclude_color is not None:
            # Convert exclude_color from normalized (0–1) to 0–255
            target = tuple(int(c * 255) for c in exclude_color)
            filtered_pixels = [
                p for p in pixel_tuples
                if not (
                    abs(p[0] - target[0]) < tolerance and
                    abs(p[1] - target[1]) < tolerance and
                    abs(p[2] - target[2]) < tolerance
                )
            ]
            if filtered_pixels:
                mode_color = Counter(filtered_pixels).most_common(1)[0][0]
            else:
                mode_color = Counter(pixel_tuples).most_common(1)[0][0]
        else:
            mode_color = Counter(pixel_tuples).most_common(1)[0][0]

        normalized_color = tuple(c / 255 for c in mode_color)
        doc.close()
        return normalized_color

    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Creates a new PDF by duplicating the original, redacting text in extracted areas,
        and inserting translated text into those areas.
        
        Expects `structured_data` to be a dict with a "paragraphs" key, e.g.:
          {
            "paragraphs": [
              {
                "text": "...",
                "bbox": [...],
                "page": 0,
                "font": "...",
                "size": 12,
                "color": 0xRRGGBB,
                "bold": bool,
                "italic": bool,
                "spacing": float,
                ...
              },
              ...
            ]
          }
        """
        doc = fitz.open(original_filepath)

        # === STEP 1: Redact original text ===
        for para in structured_data.get("paragraphs", []):
            page_num = para["page"]
            bbox = para["bbox"]
            # Slightly shrink the bbox to avoid overlapping boundaries
            # (some margin around the bounding box)
            if (bbox[2] - bbox[0] > 4) and (bbox[3] - bbox[1] > 4):
                bbox = (bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2)
            page = doc[page_num]
            redaction_rect = fitz.Rect(bbox)

            # Convert color_int to normalized RGB => exclude_color
            color_int = para["color"]
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            font_color = (r / 255, g / 255, b / 255)

            fill_color = self.get_background_color_from_bbox(
                original_filepath,
                page_num,
                bbox,
                zoom=2,
                exclude_color=font_color,
                tolerance=20
            )
            page.add_redact_annot(redaction_rect, fill=fill_color)

        # Apply redactions on each page
        for page in doc:
            page.apply_redactions()

        # === STEP 2: Insert translated text ===
        for para in structured_data.get("paragraphs", []):
            page_num = para["page"]
            page = doc[page_num]
            bbox = para["bbox"]
            if (bbox[2] - bbox[0] > 4) and (bbox[3] - bbox[1] > 4):
                bbox = (bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2)

            x0, y0, x1, y1 = bbox
            text = para["text"]
            font = para["font"]
            font_size = para["size"]
            color_int = para["color"]
            is_bold = para.get("bold", False)
            is_italic = para.get("italic", False)
            spacing = para.get("spacing", 1)

            if not text.strip():
                continue

            # Convert color int => normalized RGB
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            color_rgb = (r / 255, g / 255, b / 255)

            # Ensure valid font
            valid_fonts = ["Helvetica", "Courier"]
            if font not in valid_fonts:
                if is_bold and is_italic:
                    font = "Times-BoldItalic"
                elif is_bold:
                    font = "Times-Bold"
                elif is_italic:
                    font = "Times-Italic"
                else:
                    font = "Times-Roman"
            else:
                # If using a valid built-in font:
                if is_bold and is_italic:
                    font += "-BoldOblique"
                elif is_bold:
                    font += "-Bold"
                elif is_italic:
                    font += "-Oblique"

            # Insert text in a slightly padded rectangle
            padding = 20
            text_box_rect = fitz.Rect(x0 + padding / 4, y0, x1 + padding, y1 + padding)

            page.insert_textbox(
                text_box_rect,
                text,
                fontsize=font_size * 0.8,
                fontname=font,
                color=color_rgb,
                lineheight=spacing,
                align=0  # left aligned
            )

        # === STEP 3: Save the modified PDF ===
        doc.save(output_filepath, garbage=4, deflate=True)
        doc.close()
        print(f"✅ New PDF created from '{original_filepath}' => '{output_filepath}'")

class DOCXGenerator(BaseOutputGenerator):
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        from docx import Document
        doc = Document()
        for para in structured_data.get("paragraphs", []):
            doc.add_paragraph(para["text"])
        doc.save(output_filepath)

class HTMLGenerator(BaseOutputGenerator):
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        html = "<html><body>\n"
        for para in structured_data.get("paragraphs", []):
            html += f"<p>{para['text']}</p>\n"
        html += "</body></html>"
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(html)

class TXTGenerator(BaseOutputGenerator):
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for para in structured_data.get("paragraphs", []):
                f.write(para["text"] + "\n")

class GeneratorAgent:
    """
    Agent responsible for generating output files with translated content.
    Initialized with a file type (e.g., "pdf", "docx", "html", or "txt").
    """
    def __init__(self, file_type: str):
        self.file_type = file_type

    def generate(self, structured_data_dict: dict, original_filepath: str, output_filepath: str):
        """
        Dispatches to the appropriate generator based on self.file_type.
        
        structured_data_dict should contain a 'paragraphs' list inside, e.g.:
            {
              "paragraphs": [...]
            }
        """
        if self.file_type == "pdf":
            generator = PDFGenerator()
        elif self.file_type == "docx":
            generator = DOCXGenerator()
        elif self.file_type == "html":
            generator = HTMLGenerator()
        elif self.file_type == "txt":
            generator = TXTGenerator()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        generator.generate_output(structured_data_dict, original_filepath, output_filepath)
