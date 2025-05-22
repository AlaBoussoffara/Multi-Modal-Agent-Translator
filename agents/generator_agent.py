"""
Module pour générer des fichiers de sortie qui intègrent le contenu traduit dans la mise en page originale.

Ce module définit des classes de base abstraites et des implémentations concrètes
pour les sorties PDF, DOCX, HTML et TXT.
"""

import os
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import fitz  # PyMuPDF
from PIL import Image

from pdf2docx import Converter

class BaseOutputGenerator(ABC):
    """
    Classe de base abstraite pour les générateurs de sortie.
    """
    @abstractmethod
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Réinsère le contenu traduit dans la mise en page originale et enregistre le fichier de sortie.

        Args:
            structured_data (dict): Contenu traduit structuré par paragraphes.
            original_filepath (str): Chemin vers le fichier original.
            output_filepath (str): Chemin pour enregistrer le fichier de sortie.
        """

class PDFGenerator(BaseOutputGenerator):
    """
    Génère une sortie PDF en utilisant PyMuPDF, gère la rédaction et l'insertion du texte traduit.
    """
    def get_background_color_from_bbox(self, pdf_path, page_number, bbox,
                                       zoom=2, border_width=5, exclude_color=None,
                                       tolerance=20):
        """
        Échantillonne la couleur d'arrière-plan à partir d'une zone de boîte englobante dans une page PDF.

        Args:
            pdf_path (str): Chemin vers le fichier PDF.
            page_number (int): Numéro de la page.
            bbox (tuple): Boîte englobante (x0, y0, x1, y1).
            zoom (int, optionnel): Facteur de zoom pour l'échantillonnage.
            border_width (int, optionnel): Largeur de la bordure.
            exclude_color (tuple, optionnel): Couleur RVB à exclure.
            tolerance (int, optionnel): Tolérance pour l'exclusion de couleur.

        Returns:
            tuple: Couleur RVB normalisée.
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
        Génère un PDF en rédigeant le texte original et en insérant le texte traduit.

        Args:
            structured_data (dict): Contenu traduit avec les métadonnées des paragraphes.
            original_filepath (str): Chemin vers le PDF original.
            output_filepath (str): Chemin pour enregistrer le PDF modifié.
        """
        doc = fitz.open(original_filepath)
        
        # Etape 1: lecture du fichier original
        for para in structured_data.get("paragraphs", []):
            page_num = para["page"]
            bbox = para["bbox"]
            page = doc[page_num]
            redaction_rect = fitz.Rect(bbox)
            if (bbox[2] - bbox[0] > 4) and (bbox[3] - bbox[1] > 4):
                bbox = (bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1)
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

        # Etape 2: insertion du texte traduit
        for para in structured_data.get("paragraphs", []):
            page_num = para["page"]
            page = doc[page_num]
            bbox = para["bbox"]
            bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)
            if (bbox[2] - bbox[0] > 4) and (bbox[3] - bbox[1] > 4):
                bbox = (bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1)
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

        # Etape 3 : sauvegarde du PDF
        doc.save(output_filepath, garbage=4, deflate=True, clean=True)
        doc.close()
        print(f"PDF généré '{original_filepath}' => '{output_filepath}'")

class DOCXGenerator(BaseOutputGenerator):
    """
    Génère un fichier DOCX en convertissant un PDF généré par PDFGenerator en DOCX.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Génère un fichier DOCX en convertissant un PDF généré par PDFGenerator.

        Args:
            structured_data (dict): Contenu traduit avec les métadonnées des paragraphes.
            original_filepath (str): Chemin vers le fichier DOCX original.
            output_filepath (str): Chemin pour enregistrer le fichier DOCX modifié.
        """
        temp_pdf_path = output_filepath.replace(".docx", "_temp.pdf")
        original_filepath_pdf = original_filepath.replace(".docx", "_temp.pdf")
        pdf_generator = PDFGenerator()
        pdf_generator.generate_output(structured_data, original_filepath_pdf, temp_pdf_path)

        pdf_converter = Converter(temp_pdf_path)
        pdf_converter.convert(output_filepath, start=0, end=None)
        pdf_converter.close()

        os.remove(temp_pdf_path)
        os.remove(original_filepath_pdf)

class HTMLGenerator(BaseOutputGenerator):
    """
    Génère un fichier HTML avec le contenu traduit.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Génère un fichier HTML à partir des paragraphes traduits.

        Args:
            structured_data (dict): Contenu traduit.
            original_filepath (str): Chemin vers le fichier original.
            output_filepath (str): Chemin pour enregistrer le fichier HTML.
        """
        html = "<html><body>\n"
        for para in structured_data.get("paragraphs", []):
            html += f"<p>{para['text']}</p>\n"
        html += "</body></html>"
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(html)

class TXTGenerator(BaseOutputGenerator):
    """
    Génère un fichier texte brut avec le contenu traduit.
    """
    def generate_output(self, structured_data: dict, original_filepath: str, output_filepath: str):
        """
        Génère un fichier TXT à partir des paragraphes traduits.

        Args:
            structured_data (dict): Contenu traduit.
            original_filepath (str): Chemin vers le fichier original.
            output_filepath (str): Chemin pour enregistrer le fichier TXT.
        """
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for para in structured_data.get("paragraphs", []):
                f.write(para["text"] + "\n")

class GeneratorAgent:
    """
    Agent responsable de la génération des fichiers de sortie avec le contenu traduit selon le type de fichier.
    """
    def __init__(self, file_type: str):
        """
        Initialise l'agent générateur.

        Args:
            file_type (str): Type de fichier ('pdf', 'docx', 'html' ou 'txt').
        """
        self.file_type = file_type

    def generate(self, structured_data_dict: dict, original_filepath: str, output_filepath: str):
        """
        Génère le fichier de sortie en utilisant le générateur approprié selon le type de fichier.

        Args:
            structured_data_dict (dict): Contenu traduit structuré.
            original_filepath (str): Chemin vers le fichier original.
            output_filepath (str): Chemin pour enregistrer le fichier généré.

        Raises:
            ValueError: Si le type de fichier n'est pas supporté.
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
