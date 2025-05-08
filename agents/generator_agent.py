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

class DOCXGenerator(BaseOutputGenerator):
    """
    Generates a DOCX file with translated content and reinserts images.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Generate a DOCX document from the translated paragraphs and images.

        Args:
            structured_data (dict): Translated content.
            original_filepath (str): Path to the original file.
            output_filepath (str): Path to save the DOCX file.
        """
        doc = Document()
        paragraphs = structured_data.get("paragraphs", [])
        images = structured_data.get("images", [])

        image_index = 0
        for i, para in enumerate(paragraphs):
            # Add paragraph with styles
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(para["text"])

            # Apply styles
            run.bold = para.get("bold", False)
            run.italic = para.get("italic", False)
            font = run.font
            font.name = para.get("font", "Times New Roman")
            font.size = Pt(para.get("size", 12))
            color_int = para.get("color", 0x000000)
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            font.color.rgb = RGBColor(r, g, b)

            # Check if an image needs to be inserted at this position
            while image_index < len(images) and images[image_index]["position"] == i:
                # Ensure the image path is resolved correctly
                image_path = images[image_index]["name"]
                try:
                    doc.add_picture(image_path)
                except Exception as e:
                    print(f"⚠️ Failed to add image at position {i}: {e}")
                image_index += 1

        doc.save(output_filepath)
        print(f"✅ New DOCX created from '{original_filepath}' => '{output_filepath}'")
