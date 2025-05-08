"""
Ce module fournit des classes et des fonctions pour extraire du contenu structuré
à partir de différents types de fichiers tels que PDF, DOCX, HTML et TXT.
"""

from abc import ABC, abstractmethod
from collections import Counter
import fitz  # PyMuPDF
import numpy as np
from docx import Document
from bs4 import BeautifulSoup


def standardize_paragraph(p: dict) -> dict:
    """
    Standardise un paragraphe brut en un format structuré.

    Args:
        p (dict): Dictionnaire brut contenant les informations du paragraphe.

    Returns:
        dict: Paragraphe structuré avec des clés standardisées.
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
    Classe abstraite pour l'extraction de contenu à partir de fichiers.
    """
    @abstractmethod
    def extract_content(self, filepath: str) -> dict:
        """
        Extrait le contenu structuré d'un fichier.

        Args:
            filepath (str): Chemin vers le fichier.

        Returns:
            dict: Dictionnaire contenant une clé 'paragraphs' avec une liste de paragraphes structurés.
        """


class PDFExtractor(BaseExtractor):
    """
    Extrait le contenu des fichiers PDF en utilisant PyMuPDF.
    """
    def get_dominant_font_properties(self, paragraph_blocks: list) -> dict:
        """
        Détermine les propriétés de police les plus fréquentes dans une liste de blocs de paragraphes.

        Args:
            paragraph_blocks (list): Liste des blocs contenant des informations sur la police.

        Returns:
            dict: Propriétés dominantes de la police.
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
        Extrait le contenu structuré d'un fichier PDF.

        Args:
            filepath (str): Chemin vers le fichier PDF.
            tolerance_factor (float, optionnel): Facteur pour déterminer la séparation des paragraphes.

        Returns:
            dict: Dictionnaire contenant des paragraphes structurés sous la clé 'paragraphs'.
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

            # Regroupe les lignes en paragraphes
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
                    or (vertical_gap > tolerance_factor)
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
    Extrait le contenu des fichiers DOCX en utilisant python-docx.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extrait le contenu structuré d'un fichier DOCX, y compris les images.

        Args:
            filepath (str): Chemin vers le fichier DOCX.

        Returns:
            dict: Dictionnaire contenant des paragraphes structurés sous la clé 'paragraphs' et des images sous la clé 'images'.
        """
        print(f"[DOCXExtractor] Extraction du contenu de {filepath}...")
        document = Document(filepath)
        paragraphs = []
        images = []
        image_counter = 0

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

        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                image_counter += 1
                image_data = rel.target_part.blob
                image_name = f"image_{image_counter}.png"
                with open(image_name, "wb") as img_file:
                    img_file.write(image_data)
                images.append({"name": image_name, "position": len(paragraphs)})

        return {"paragraphs": [standardize_paragraph(p) for p in paragraphs], "images": images}


class HTMLExtractor(BaseExtractor):
    """
    Extrait le contenu des fichiers HTML en utilisant BeautifulSoup.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extrait le contenu structuré d'un fichier HTML.

        Args:
            filepath (str): Chemin vers le fichier HTML.

        Returns:
            dict: Dictionnaire contenant des paragraphes structurés sous la clé 'paragraphs'.
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
    Extrait le contenu des fichiers texte brut.
    """
    def extract_content(self, filepath: str) -> dict:
        """
        Extrait le contenu structuré d'un fichier TXT.

        Args:
            filepath (str): Chemin vers le fichier texte.

        Returns:
            dict: Dictionnaire contenant un paragraphe structuré sous la clé 'paragraphs'.
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
    Agent pour extraire le contenu structuré des fichiers en fonction de leur type.
    """
    def __init__(self, file_type: str):
        """
        Initialise l'agent d'extraction.

        Args:
            file_type (str): Type du fichier ('pdf', 'docx', 'html', ou 'txt').
        """
        self.file_type = file_type

    def extract(self, filepath: str) -> dict:
        """
        Extrait le contenu structuré du fichier donné.

        Args:
            filepath (str): Chemin vers le fichier.

        Returns:
            dict: Dictionnaire contenant des paragraphes structurés.

        Raises:
            ValueError: Si le type de fichier n'est pas pris en charge.
        """
        if self.file_type == "pdf":
            extractor = PDFExtractor()
            return extractor.extract_content(filepath)
        elif self.file_type == "docx":
            extractor = DOCXExtractor()
            return extractor.extract_content(filepath)
        elif self.file_type == "html":
            extractor = HTMLExtractor()
        elif self.file_type == "txt":
            extractor = TXTExtractor()
        else:
            raise ValueError(f"Type de fichier non pris en charge : {self.file_type}")
