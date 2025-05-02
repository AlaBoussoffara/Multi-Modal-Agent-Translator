import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer, util
import re  # Importer le module pour les expressions régulières
import pickle

# Chemin vers le fichier PDF
pdf_path = "src_documents/dictgeniecivil.pdf"

# Vérifier si le fichier existe
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Le fichier PDF '{pdf_path}' est introuvable. Vérifiez le chemin.")

# Charger le PDF
doc = fitz.open(pdf_path)
glossary = []

# Liste des indications à supprimer
glossary_indications = ["m", "m,", "f", "f,", "adj", "adj,", "vb", "vb,"]

for page_num, page in enumerate(doc):
    if not (258 <= page_num and page_num <= 444):
        continue
    # print("Page", page_num)
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

                # Supprimer uniquement les indications listées, sans affecter le reste du texte
                text = re.sub(rf'\b({"|".join(re.escape(ind) for ind in glossary_indications)})\b', '', text).strip()
                if not text:  # Si le texte est vide après nettoyage, passer au suivant
                    continue

                # Vérifier si le texte est en gras (bold)
                if "bold" in span.get("font", "").lower():
                    if term_translated:  # Si un terme traduit est déjà accumulé, ignorer
                        continue
                    term_original = f"{term_original} {text}".strip()  # Ajouter au terme original avec un espace
                else:
                    term_translated = f"{term_translated} {text}".strip()  # Ajouter au terme traduit avec un espace

            # Réorganiser les termes originaux si une virgule est présente
            if term_original and "," in term_original:
                parts = [part.strip() for part in term_original.split(",")]
                if len(parts) == 2:
                    term_original = f"{parts[1]} {parts[0]}"

            # Si les deux termes sont trouvés, les ajouter à la liste structurée
            if term_original and term_translated:
                glossary.append((term_original, term_translated))
                # print(f"Original: {term_original} ; Translated: {term_translated}")

# Afficher les données structurées
# print("\nExtracted Terms:")
# for original, translated in glossary:
#         print(f"{original} -> {translated}")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Générer les embeddings pour le glossaire
glossary_embeddings = []
length = len(glossary)
i = 0
for original, translated in glossary:
    embedding = model.encode(original, convert_to_tensor=True)
    glossary_embeddings.append((original, translated, embedding))
    # print(f"Original: {original} ; Translated: {translated}")
    i += 1
    if i % 100 == 0:  # Afficher la progression tous les 100 termes
        print(f"Progress: {i}/{length} ({(i/length)*100:.2f}%)")

with open("glossary_embeddings.pkl", "wb") as f:
    pickle.dump(glossary_embeddings, f)

print("Embeddings sauvegardés dans 'glossary_embeddings.pkl'.")

# # Texte source à traduire
# source_text = "The dissolved acetylene is highly reactive."
# source_embedding = model.encode(source_text, convert_to_tensor=True)

# # Trouver les termes pertinents
# relevant_terms = []
# for original, translated, embedding in glossary_embeddings:
#     similarity = util.cos_sim(source_embedding, embedding).item()
#     if similarity > 0.5:  # Seuil de similarité (ajustez selon vos besoins)
#         relevant_terms.append((original, translated, similarity))

# # Trier les termes par pertinence (similarité décroissante)
# relevant_terms = sorted(relevant_terms, key=lambda x: x[2], reverse=True)
# print(relevant_terms)