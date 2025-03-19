# extractor_agent.py

from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import numpy as np
from collections import Counter

class BaseExtractor(ABC):
    @abstractmethod
    def extract_content(self, filepath: str) -> dict:
        """
        Extracts structured content (e.g., paragraphs with text and formatting)
        from a file and returns a standardized dictionary with a 'paragraphs' key.
        """
        pass

class PDFExtractor(BaseExtractor):
    """
    Extractor for PDF files using PyMuPDF. Groups text blocks into paragraphs based on spacing,
    font differences, and bounding box changes, preserving structure and layout.
    """

    def get_dominant_font_properties(self, paragraph_blocks: list[dict]) -> dict:
        """
        Determines the most frequent font properties in a paragraph block list.

        Args:
            paragraph_blocks (list of dicts): each block with keys like:
                'font', 'size', 'color', 'bold', 'italic', 'char_spacing'.

        Returns:
            dict: The dominant properties, including 'font', 'size', 'color', 'bold', 'italic',
                  'char_spacing'.
        """
        def most_frequent(items):
            return Counter(items).most_common(1)[0][0] if items else None

        fonts = [block["font"] for block in paragraph_blocks]
        sizes = [block["size"] for block in paragraph_blocks]
        colors = [block["color"] for block in paragraph_blocks]
        bold_flags = [block["bold"] for block in paragraph_blocks]
        italic_flags = [block["italic"] for block in paragraph_blocks]
        char_spacings = [block["char_spacing"] for block in paragraph_blocks]

        return {
            "font": most_frequent(fonts),
            "size": min(sizes),  # Use the smallest size among blocks
            "color": most_frequent(colors),
            "bold": most_frequent(bold_flags),
            "italic": most_frequent(italic_flags),
            "char_spacing": np.mean(char_spacings),
        }

    def extract_content(self, pdf_path: str, tolerance_factor: float = 2.5) -> dict:
        """
        Extracts paragraphs from a PDF while preserving structure and layout.

        - Processes text blocks from the PDF
        - Groups them into paragraphs based on spacing, font properties, bounding boxes, etc.

        Args:
            pdf_path (str): Path to the input PDF
            tolerance_factor (float): Factor to adjust spacing threshold

        Returns:
            dict with:
              "paragraphs": a list of paragraph dicts (keys like "text", "bbox", "font", etc.)
        """
        doc = fitz.open(pdf_path)
        structured_data = []

        for page_num, page in enumerate(doc):
            raw_blocks = []
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    text = " ".join(span["text"] for span in spans).strip()
                    if not text:
                        continue

                    font = spans[0]["font"]
                    font_size = spans[0]["size"]
                    color = spans[0]["color"]
                    flags = spans[0]["flags"]
                    char_spacing = spans[0].get("char_spacing", 0)

                    is_bold = bool(flags & 8)
                    is_italic = bool(flags & 4)
                    bbox = line["bbox"]

                    raw_blocks.append({
                        "text": text,
                        "bbox": bbox,
                        "page": page_num,
                        "font": font,
                        "size": font_size,
                        "color": color,
                        "bold": is_bold,
                        "italic": is_italic,
                        "char_spacing": char_spacing,
                    })

            # Determine an automatic spacing threshold
            line_gaps = [
                raw_blocks[i+1]["bbox"][1] - raw_blocks[i]["bbox"][3]
                for i in range(len(raw_blocks) - 1)
            ]
            auto_threshold = (min(line_gaps) * tolerance_factor) if line_gaps else 10

            paragraphs = []
            current_paragraph = None
            paragraph_properties = []

            for i, block in enumerate(raw_blocks):
                if i == 0:
                    # Start first paragraph
                    current_paragraph = {
                        "text": block["text"],
                        "bbox": block["bbox"],
                        "page": block["page"],
                        "font": block["font"],
                        "size": block["size"],
                        "color": block["color"],
                        "bold": block["bold"],
                        "italic": block["italic"],
                        "char_spacing": block["char_spacing"],
                        "spacing": 1,
                    }
                    paragraph_properties.append(block)
                    continue

                prev_bottom = current_paragraph["bbox"][3]
                curr_top = block["bbox"][1]
                vertical_gap = abs(prev_bottom - curr_top)

                if (
                    (current_paragraph["font"] != block["font"])
                    or (abs(current_paragraph["size"] - block["size"]) > 1)
                    or (vertical_gap > auto_threshold)
                ):
                    # Finalize current paragraph
                    dominant_props = self.get_dominant_font_properties(paragraph_properties)
                    current_paragraph.update(dominant_props)
                    paragraphs.append(current_paragraph)

                    # Start a new paragraph
                    current_paragraph = {
                        "text": block["text"],
                        "bbox": block["bbox"],
                        "page": block["page"],
                        "font": block["font"],
                        "size": block["size"],
                        "color": block["color"],
                        "bold": block["bold"],
                        "italic": block["italic"],
                        "char_spacing": block["char_spacing"],
                        "spacing": 1,
                    }
                    paragraph_properties = [block]
                else:
                    # Continue the same paragraph
                    current_paragraph["spacing"] = max(
                        1.75 * vertical_gap / block["size"],
                        current_paragraph["spacing"],
                    )
                    current_paragraph["text"] += " " + block["text"]
                    current_paragraph["bbox"] = (
                        min(current_paragraph["bbox"][0], block["bbox"][0]),
                        current_paragraph["bbox"][1],
                        max(block["bbox"][2], current_paragraph["bbox"][2]),
                        block["bbox"][3],
                    )
                    paragraph_properties.append(block)

            # Final paragraph in this page
            if current_paragraph:
                dominant_props = self.get_dominant_font_properties(paragraph_properties)
                current_paragraph.update(dominant_props)
                paragraphs.append(current_paragraph)

            structured_data.extend(paragraphs)

        doc.close()
        return {"paragraphs": structured_data}

class DOCXExtractor(BaseExtractor):
    def extract_content(self, filepath: str) -> dict:
        import docx
        document = docx.Document(filepath)
        paragraphs = []
        for para in document.paragraphs:
            if para.text.strip():
                paragraphs.append({
                    "text": para.text.strip(),
                    "bbox": (0, 0, 100, 20),
                    "page": 0,
                    "font": "Times-Roman",
                    "size": 12,
                    "color": 0x000000,
                    "bold": para.runs[0].bold if para.runs else False,
                    "italic": para.runs[0].italic if para.runs else False,
                })
        return {"paragraphs": paragraphs}

class HTMLExtractor(BaseExtractor):
    def extract_content(self, filepath: str) -> dict:
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text:
                paragraphs.append({
                    "text": text,
                    "bbox": (0, 0, 100, 20),
                    "page": 0,
                    "font": "Times-Roman",
                    "size": 12,
                    "color": 0x000000,
                    "bold": False,
                    "italic": False,
                })
        return {"paragraphs": paragraphs}

class TXTExtractor(BaseExtractor):
    def extract_content(self, filepath: str) -> dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return {"paragraphs": [{
            "text": text,
            "bbox": (0, 0, 100, 20),
            "page": 0,
            "font": "Times-Roman",
            "size": 12,
            "color": 0x000000,
            "bold": False,
            "italic": False,
        }]}

class ExtractorAgent:
    """
    Agent that uses a file_type provided at initialization
    to dispatch to the correct extractor.
    """
    def __init__(self, file_type: str):
        """
        Args:
            file_type (str): The file type (pdf, docx, html, txt, etc.).
        """
        self.file_type = file_type

    def extract(self, filepath: str) -> dict:
        """
        Dispatch to the correct extractor based on the stored file_type.
        """
        if self.file_type == "pdf":
            extractor = PDFExtractor()
            return extractor.extract_content(filepath, tolerance_factor=2.5)
        elif self.file_type == "docx":
            extractor = DOCXExtractor()
        elif self.file_type == "html":
            extractor = HTMLExtractor()
        elif self.file_type == "txt":
            extractor = TXTExtractor()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        return extractor.extract_content(filepath)
