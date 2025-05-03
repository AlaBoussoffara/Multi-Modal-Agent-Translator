import fitz  # PyMuPDF
import os
import re
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Chemin vers le fichier PDF
PDF_PATH = "glossary/dictgeniecivil.pdf"

# Chemins pour sauvegarder les glossaires
GLOSSARY_ENGFR_PATH = "glossary/glossary_engfr_embeddings.pickle"
GLOSSARY_FRENG_PATH = "glossary/glossary_freng_embeddings.pickle"

# Liste des indications à supprimer
GLOSSARY_INDICATIONS = ["m", "m,", "f", "f,", "adj", "adj,", "vb", "vb,"]

def extract_glossary(doc, start_page, end_page):
    """
    Extrait les termes du glossaire d'un PDF entre deux pages données.

    Args:
        doc (fitz.Document): Document PDF chargé avec PyMuPDF.
        start_page (int): Numéro de la première page à traiter.
        end_page (int): Numéro de la dernière page à traiter.

    Returns:
        list: Liste de tuples (terme_original, terme_traduit).
    """
    glossary = []

    for page_num, page in enumerate(doc):
        if not (start_page <= page_num <= end_page):
            continue

        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Initialiser les variables pour stocker les termes
                term_original = ""
                term_translated = ""

                for span in spans:
                    text = span["text"].strip()
                    if not text:
                        continue

                    # Supprimer uniquement les indications listées
                    text = re.sub(rf'\b({"|".join(re.escape(ind) for ind in GLOSSARY_INDICATIONS)})\b', '', text).strip()
                    if not text:
                        continue

                    # Vérifier si le texte est en gras (bold)
                    if "bold" in span.get("font", "").lower():
                        if term_translated:  # Si un terme traduit est déjà accumulé, ignorer
                            continue
                        term_original = f"{term_original} {text}".strip()
                    else:
                        term_translated = f"{term_translated} {text}".strip()

                # Réorganiser les termes originaux si une virgule est présente
                if term_original and "," in term_original:
                    parts = [part.strip() for part in term_original.split(",")]
                    if len(parts) == 2:
                        term_original = f"{parts[1]} {parts[0]}"

                # Ajouter les termes au glossaire
                if term_original and term_translated:
                    glossary.append((term_original, term_translated))

    return glossary

def generate_embeddings(glossary, model, output_path):
    """
    Génère les embeddings pour un glossaire et les sauvegarde dans un fichier.

    Args:
        glossary (list): Liste de tuples (terme_original, terme_traduit).
        model (SentenceTransformer): Modèle pour générer les embeddings.
        output_path (str): Chemin pour sauvegarder les embeddings.
    """
    glossary_embeddings = []

    # Utiliser tqdm pour afficher la progression
    with tqdm(total=len(glossary), desc="Génération des embeddings", unit="terme") as pbar:
        for original, translated in glossary:
            embedding = model.encode(original, convert_to_tensor=True)
            glossary_embeddings.append((original, translated, embedding))
            pbar.update(1)  # Mettre à jour la barre de progression

    # Sauvegarder les embeddings dans un fichier
    with open(output_path, "wb") as f:
        pickle.dump(glossary_embeddings, f)
    print(f"Embeddings sauvegardés dans '{output_path}'.")

def main():
    # Vérifier si le fichier PDF existe
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Le fichier PDF '{PDF_PATH}' est introuvable. Vérifiez le chemin.")

    # Charger le PDF
    doc = fitz.open(PDF_PATH)

    # Charger le modèle SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extraire et sauvegarder le glossaire anglais -> français
    print("Extraction du glossaire anglais -> français...")
    glossary_engfr = extract_glossary(doc, start_page=22, end_page=254)
    print(f"Glossaire anglais -> français extrait : {len(glossary_engfr)} termes.")
    generate_embeddings(glossary_engfr, model, GLOSSARY_ENGFR_PATH)

    # Extraire et sauvegarder le glossaire français -> anglais
    print("Extraction du glossaire français -> anglais...")
    glossary_freng = extract_glossary(doc, start_page=258, end_page=444)
    print(f"Glossaire français -> anglais extrait : {len(glossary_freng)} termes.")
    generate_embeddings(glossary_freng, model, GLOSSARY_FRENG_PATH)

if __name__ == "__main__":
    main()