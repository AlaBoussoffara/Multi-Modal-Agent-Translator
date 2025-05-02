"""
Module contenant l'agent d'évaluation (EvaluatorAgent).

L'agent d'évaluation utilise le modèle COMET pour calculer des scores de qualité de traduction
en se basant sur des triplets standardisés : texte original, texte traduit, et texte de référence.
"""

from comet import download_model, load_from_checkpoint
import pandas as pd
import os
# from PyDeepLX import PyDeepLX

class EvaluatorAgent:
    """
    Classe pour évaluer la qualité des traductions à l'aide de COMET.

    Les évaluations sont basées sur des triplets contenant :
    - 'filename' : Nom du fichier.
    - 'src' : Texte original.
    - 'mt' : Texte traduit par la machine.
    - 'ref' : Texte de référence.
    """
    def __init__(self, model_path, evaluation_results_path="comet_scores"):
        """
        Initialise l'agent d'évaluation avec le modèle COMET.

        Args:
            model_path (str): Chemin vers le modèle COMET.
            evaluation_results_path (str, optionnel): Répertoire pour sauvegarder les résultats. Par défaut : "comet_scores".
        """
        self.model = load_from_checkpoint(model_path)  # Charge le modèle COMET
        self.evaluation_results_path = evaluation_results_path  # Répertoire pour les résultats

    def evaluate_triplets(self, triplets):
        """
        Évalue une liste de triplets et retourne les scores COMET.

        Args:
            triplets (list): Liste de dictionnaires contenant 'filename', 'src', 'mt', et 'ref'.

        Returns:
            list: Résultats de l'évaluation avec les scores COMET.
        """
        # Prépare les données pour le modèle COMET
        data = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triplets]
        
        # Prédit les scores COMET
        scores = self.model.predict(data)["scores"]
        
        # Associe les scores aux fichiers correspondants
        results = [{"filename": t["filename"], "COMET Score": score} for t, score in zip(triplets, scores)]
        
        # Sauvegarde les résultats dans un fichier CSV
        os.makedirs(self.evaluation_results_path, exist_ok=True)  # Crée le répertoire si nécessaire
        output_file = os.path.join(self.evaluation_results_path, "evaluation_results.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Évaluation terminée ! Résultats sauvegardés dans '{output_file}'")
        
        return results  # Retourne les résultats


# print(PyDeepLX.translate('ceci est un test'))