"""
This module provides classes and functions for extracting structured content 
from various file types such as PDF, DOCX, HTML, and TXT.
"""

from abc import ABC, abstractmethod
from collections import Counter
import fitz  # PyMuPDF
import numpy as np
from docx import Document
from bs4 import BeautifulSoup


def standardize_paragraph(p: dict) -> dict:
    """
    Convert a raw paragraph dictionary into a standardized schema.

    Expected keys:
      - text: paragraph text
      - bbox: tuple (x0, y0, x1, y1) for location (default for non-PDF)
      - page: page number (default 0)
      - font: font name (default "Times-Roman")
      - size: font size (default 12)
      - color: color as an integer (default 0x000000)
      - bold: boolean for bold formatting (default False)
      - italic: boolean for italic formatting (default False)
      - spacing: line/paragraph spacing (default 1)
      - raw_metadata: any additional fields from extraction output

    Args:
        p (dict): Raw paragraph dictionary.

    Returns:
        dict: Standardized paragraph dictionary.
    """
    standardized = {}
    standardized["text"] = p.get("text", "")
    standardized["bbox"] = p.get("bbox", (0, 0, 100, 20))
    standardized["page"] = p.get("page", 0)
    standardized["font"] = p.get("font", "Times-Roman")
    standardized["size"] = p.get("size", 12)
    standardized["color"] = p.get("color", 0x000000)
    standardized["bold"] = p.get("bold", False)
    standardized["italic"] = p.get("italic", False)
    standardized["spacing"] = p.get("spacing", 1)
    keys = {"text", "bbox", "page", "font", "size", "color", "bold", "italic", "spacing"}
    standardized["raw_metadata"] = {k: v for k, v in p.items() if k not in keys}
    return standardized

class BaseExtractor(ABC):
    """
    Abstract base class for content extraction from files.
    """
    @abstractmethod
    def extract_content(self, filepath: str) -> dict:
        """
        Extract structured content from a file and return a standardized dictionary 
        containing a 'paragraphs' key with a list of standardized paragraph dictionaries.

        Args:
            filepath (str): Path to the file.

        Returns:
            dict: Dictionary with key 'paragraphs'.
        """

class PDFExtractor(BaseExtractor):
    """
    Extracts content from PDF files using PyMuPDF.
    """
    def get_dominant_font_properties(self, paragraph_blocks: list) -> dict:
        """
        Determine the most frequent font properties from a list of paragraph blocks.

        Args:
            paragraph_blocks (list): List of blocks containing font information.

        Returns:
            dict: Dominant font properties.
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
            "size": min(sizes),
            "color": most_frequent(colors),
            "bold": most_frequent(bold_flags),
            "italic": most_frequent(italic_flags),
            "char_spacing": np.mean(char_spacings),
        }

    def extract_content(self, filepath: str, tolerance_factor: float = 2.5) -> dict:
        """
        Extract structured content from a PDF file.

        Args:
            filepath (str): Path to the PDF file.
            tolerance_factor (float, optional): Factor to determine paragraph separation.
            Default is 2.5.

        Returns:
            dict: Dictionary containing standardized paragraphs under the 'paragraphs' key.
        """
        doc = fitz.open(filepath)
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
                    dominant_props = self.get_dominant_font_properties(paragraph_properties)
                    current_paragraph.update(dominant_props)
                    paragraphs.append(current_paragraph)

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

            if current_paragraph:
                dominant_props = self.get_dominant_font_properties(paragraph_properties)
                current_paragraph.update(dominant_props)
                paragraphs.append(current_paragraph)

            structured_data.extend(paragraphs)

        doc.close()
        standardized = [standardize_paragraph(p) for p in structured_data]
        return {"paragraphs": standardized}

class DOCXExtractor(BaseExtractor):
    """
    Extracts content from DOCX files using python-docx.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extract content from a DOCX file.

        Args:
            filepath (str): Path to the DOCX file.

        Returns:
            dict: Dictionary containing standardized paragraphs under the 'paragraphs' key.
        """
        document = Document(filepath)
        paragraphs = []
        for para in document.paragraphs:
            if not para.runs:
                continue
            merged_text = ""
            current_text = ""
            current_style = (
                para.runs[0].bold,
                para.runs[0].italic,
                para.runs[0].underline,
                para.runs[0].font.name,
                para.runs[0].font.size,
                para.runs[0].font.color.rgb
            )
            for run in para.runs:
                style = (
                    run.bold,
                    run.italic,
                    run.underline,
                    run.font.name,
                    run.font.size,
                    run.font.color.rgb
                )
                if style == current_style:
                    current_text += run.text
                else:
                    merged_text += current_text
                    current_text = run.text
                    current_style = style
            merged_text += current_text
            paragraphs.append({
                "text": merged_text,
                "bbox": (0, 0, 100, 20),
                "page": 0,
                "font": "Times-Roman",
                "size": 12,
                "color": 0x000000,
                "bold": para.runs[0].bold if para.runs else False,
                "italic": para.runs[0].italic if para.runs else False,
            })
        return {"paragraphs": [standardize_paragraph(p) for p in paragraphs]}

class HTMLExtractor(BaseExtractor):
    """
    Extracts content from HTML files using BeautifulSoup.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extract content from an HTML file.

        Args:
            filepath (str): Path to the HTML file.

        Returns:
            dict: Dictionary containing standardized paragraphs under the 'paragraphs' key.
        """
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
        return {"paragraphs": [standardize_paragraph(p) for p in paragraphs]}

class TXTExtractor(BaseExtractor):
    """
    Extracts content from plain text files.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extract content from a TXT file.

        Args:
            filepath (str): Path to the text file.

        Returns:
            dict: Dictionary containing a single standardized paragraph under the 'paragraphs' key.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return {"paragraphs": [standardize_paragraph({
            "text": text,
            "bbox": (0, 0, 100, 20),
            "page": 0,
            "font": "Times-Roman",
            "size": 12,
            "color": 0x000000,
            "bold": False,
            "italic": False,
        })]}

class ExtractorAgent:
    """
    Agent for extracting structured content from files based on file type.
    """
    def __init__(self, file_type: str):
        """
        Initialize the extractor agent.

        Args:
            file_type (str): Type of the file ('pdf', 'docx', 'html', or 'txt').
        """
        self.file_type = file_type

    def extract(self, filepath: str) -> dict:
        """
        Extract structured content from the given file.

        Args:
            filepath (str): Path to the file.

        Returns:
            dict: Dictionary with a key 'paragraphs' containing standardized paragraphs.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if self.file_type == "pdf":
            extractor = PDFExtractor()
            return extractor.extract_content(filepath)
        elif self.file_type == "docx":
            extractor = DOCXExtractor()
        elif self.file_type == "html":
            extractor = HTMLExtractor()
        elif self.file_type == "txt":
            extractor = TXTExtractor()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        return extractor.extract_content(filepath)
