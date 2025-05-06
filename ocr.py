#!/usr/bin/env python3
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageStat
import pytesseract

# Spécifiez le chemin vers Tesseract si nécessaire
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Exemple pour Linux

from langchain_aws import ChatBedrock
from agents.translator_agent import TranslatorAgent

llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0}
)
translator = TranslatorAgent(llm, "french")

# Remplacez cette fonction par votre propre implémentation de traduction
def translate(text):
    translated_paragraphs = translator.translate([{"text": text}], use_glossary=False)[0]['text']
    return translated_paragraphs

def get_background_color(image, box):
    """
    Extrait la couleur moyenne (médiane) de la région définie par box (x0, y0, x1, y1).
    """
    region = image.crop(box)
    stat = ImageStat.Stat(region)
    median = stat.median  # [R, G, B]
    return tuple(int(v) for v in median)


def get_font_and_size(draw, text, box_size, font_path):
    """
    Trouve la taille de police maximale pour que `text` tienne dans `box_size`.
    box_size: (max_width, max_height)
    """
    max_width, max_height = box_size
    font_size = max_height
    while font_size > 0:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            # Si le fichier de police n'est pas trouvé, on charge la police par défaut
            return ImageFont.load_default()
        # Utilise textbbox pour calculer la taille du texte
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= max_width and h <= max_height:
            return font
        font_size -= 1
    return ImageFont.load_default()


def choose_contrast_color(bg_color):
    """
    Choisit noir ou blanc selon la luminance du fond pour garantir un bon contraste.
    """
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def process_image(input_path, output_path, font_path):
    # Ouvre l'image et convertit en RGB
    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Utilise pytesseract pour obtenir les données OCR
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])

    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text:
            continue

        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        box = (x, y, x + w, y + h)

        # 1. Récupère la couleur de fond
        bg_color = get_background_color(image, box)

        # 2. Masque l'ancien texte
        draw.rectangle(box, fill=bg_color)

        # 3. Traduit le texte
        translated = translate(text)

        # 4. Choisit la police adaptée
        font = get_font_and_size(draw, translated, (w, h), font_path)

        # 5. Détermine la couleur du texte pour un contraste optimal
        font_color = choose_contrast_color(bg_color)

        # 6. Dessine le texte traduit
        draw.text((x, y), translated, fill=font_color, font=font)

    # Sauvegarde l'image résultante
    image.save(output_path)


if __name__ == "__main__":
    process_image("src_documents/data_visualization.png", "mt_outputs/data_visualisation_translated.png", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
