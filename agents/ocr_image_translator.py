from PIL import Image, ImageDraw, ImageFont
import pytesseract

def translate_and_rewrite_image(image_path, translations, output_path):
    """
    Traduit et réécrit les termes sur une image.

    Args:
        image_path (str): Chemin de l'image originale.
        translations (dict): Dictionnaire des traductions {texte_original: texte_traduit}.
        output_path (str): Chemin pour sauvegarder l'image modifiée.
    """
    # Charger l'image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Détecter les blocs de texte avec pytesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data['text'])

    for i in range(n):
        text = data['text'][i].strip()
        if text and text in translations:
            # Obtenir les coordonnées du bloc
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Masquer le texte original (rectangle blanc)
            draw.rectangle([x, y, x + w, y + h], fill="white")

            # Écrire le texte traduit
            translated_text = translations[text]
            font = ImageFont.load_default()  # Charger une police par défaut
            draw.text((x, y), translated_text, fill="black", font=font)

    # Sauvegarder l'image modifiée
    image.save(output_path)
    print(f"✅ Image modifiée sauvegardée : {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "example_image.jpg"
    output_path = "translated_image.jpg"
    translations = {
        "Hello": "Bonjour",
        "World": "Monde"
    }
    translate_and_rewrite_image(image_path, translations, output_path)
