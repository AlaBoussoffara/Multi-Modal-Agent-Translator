"""
Module for generating output files that integrate translated content into the original layout.

This module defines abstract base classes and concrete implementations
for PDF, DOCX, HTML, and TXT outputs.
"""

from abc import ABC, abstractmethod
from collections import Counter
import fitz
import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor
from PIL import Image
import os
import tempfile
import shutil

# Convert PDF back to DOCX using pdf2docx
from pdf2docx import Converter


class BaseOutputGenerator(ABC):
    """
    Abstract base class for output generators.
    """
    @abstractmethod
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Reinserts translated content into the original layout and saves the output file.

        Args:
            structured_data (dict): Translated content structured by paragraphs.
            original_filepath (str): Path to the original file.
            output_filepath (str): Path to save the output file.
        """

class PDFGenerator(BaseOutputGenerator):
    """
    Generates PDF output using PyMuPDF, handling redaction and insertion of translated text.
    """
    def get_background_color_from_bbox(self, pdf_path, page_number, bbox,
                                       zoom=2, border_width=5, exclude_color=None,
                                       tolerance=20):
        """
        Sample the background color from a bounding box area in a PDF page.

        Args:
            pdf_path (str): Path to the PDF file.
            page_number (int): Page number.
            bbox (tuple): Bounding box (x0, y0, x1, y1).
            zoom (int, optional): Zoom factor for sampling.
            border_width (int, optional): Width of the border area.
            exclude_color (tuple, optional): RGB color to exclude.
            tolerance (int, optional): Tolerance for color exclusion.

        Returns:
            tuple: Normalized RGB color.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        x0, y0, x1, y1 = [int(coord * zoom) for coord in bbox]
        cropped = img.crop((x0, y0, x1, y1))
        arr = np.array(cropped)
        height, _ , _ = arr.shape
        border_pixels = []
        border_pixels.append(arr[0:border_width, :, :].reshape(-1, 3))
        border_pixels.append(arr[-border_width:, :, :].reshape(-1, 3))
        if height > 2 * border_width:
            border_pixels.append(arr[border_width:height-border_width, 0:border_width, :].reshape(-1, 3))
            border_pixels.append(arr[border_width:height-border_width, -border_width:, :].reshape(-1, 3))
        border_pixels = np.concatenate(border_pixels, axis=0)
        pixel_tuples = [tuple(pixel) for pixel in border_pixels]
        if exclude_color is not None:
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
        Generate a PDF by redacting original text and inserting translated text.

        Args:
            structured_data (dict): Translated content with paragraph metadata.
            original_filepath (str): Path to the original PDF.
            output_filepath (str): Path to save the modified PDF.
        """
        doc = fitz.open(original_filepath)
        # Step 1: Redact original text
        for para in structured_data.get("paragraphs", []):
            page_num = para["page"]
            bbox = para["bbox"]
            page = doc[page_num]
            redaction_rect = fitz.Rect(bbox)
            if (bbox[2] - bbox[0] > 4) and (bbox[3] - bbox[1] > 4):
                bbox = (bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2)
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
        for page in doc:
            page.apply_redactions()
        # Step 2: Insert translated text
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
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            color_rgb = (r / 255, g / 255, b / 255)
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
                if is_bold and is_italic:
                    font += "-BoldOblique"
                elif is_bold:
                    font += "-Bold"
                elif is_italic:
                    font += "-Oblique"
            padding = 20
            text_box_rect = fitz.Rect(x0 + padding / 4, y0, x1 + padding, y1 + padding)
            page.insert_textbox(
                text_box_rect,
                text,
                fontsize=font_size * 0.8,
                fontname=font,
                color=color_rgb,
                lineheight=spacing,
                align=0
            )
        # Step 3: Save the modified PDF with compression
        doc.save(output_filepath, garbage=4, deflate=True, clean=True)
        doc.close()
        print(f"✅ New PDF created from '{original_filepath}' => '{output_filepath}'")

class DOCXGenerator(BaseOutputGenerator):
    """
    Generates a DOCX file by converting a PDF generated by PDFGenerator back to DOCX.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Generate a DOCX file by converting a PDF generated by PDFGenerator.

        Args:
            structured_data (dict): Translated content with paragraph metadata.
            original_filepath (str): Path to the original DOCX file.
            output_filepath (str): Path to save the modified DOCX file.
        """
        # Generate PDF using PDFGenerator
        temp_pdf_path = output_filepath.replace(".docx", "_temp.pdf")
        original_filepath_pdf = original_filepath.replace(".docx", "_temp.pdf")
        pdf_generator = PDFGenerator()
        pdf_generator.generate_output(structured_data, original_filepath_pdf, temp_pdf_path)



        pdf_converter = Converter(temp_pdf_path)
        pdf_converter.convert(output_filepath, start=0, end=None)  # Convert all pages
        pdf_converter.close()

        # Clean up temporary translated PDF
        os.remove(temp_pdf_path)
        #clen up temporary original PDF
        os.remove(original_filepath_pdf)

class HTMLGenerator(BaseOutputGenerator):
    """
    Generates an HTML file with translated content.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Generate an HTML file from the translated paragraphs.

        Args:
            structured_data (dict): Translated content.
            original_filepath (str): Path to the original file.
            output_filepath (str): Path to save the HTML file.
        """
        html = "<html><body>\n"
        for para in structured_data.get("paragraphs", []):
            html += f"<p>{para['text']}</p>\n"
        html += "</body></html>"
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(html)

class TXTGenerator(BaseOutputGenerator):
    """
    Generates a plain text file with translated content.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Generate a TXT file from the translated paragraphs.

        Args:
            structured_data (dict): Translated content.
            original_filepath (str): Path to the original file.
            output_filepath (str): Path to save the TXT file.
        """
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for para in structured_data.get("paragraphs", []):
                f.write(para["text"] + "\n")

class GeneratorAgent:
    """
    Agent responsible for generating output files with the translated content based on file type.
    """
    def __init__(self, file_type: str):
        """
        Initialize the generator agent.

        Args:
            file_type (str): Type of the file ('pdf', 'docx', 'html', or 'txt').
        """
        self.file_type = file_type

    def generate(self, structured_data_dict: dict, original_filepath: str, output_filepath: str):
        """
        Generate the output file using the appropriate generator based on the file type.

        Args:
            structured_data_dict (dict): Structured translated content.
            original_filepath (str): Path to the original file.
            output_filepath (str): Path to save the generated file.

        Raises:
            ValueError: If the file type is unsupported.
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
