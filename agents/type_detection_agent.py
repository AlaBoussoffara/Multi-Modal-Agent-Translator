"""
Module pour détecter le type de fichier en fonction de son extension.

Classes :
    TypeDetectionAgent : Classe pour détecter le type de fichier.
Fonctions :
    detect_file_type(filepath: str) -> str : Fonction utilitaire pour détecter le type de fichier.
"""
import os

class TypeDetectionAgent:
    """
    Agent pour détecter le type de fichier en fonction de son extension.

    Méthodes :
        detect(filepath: str) -> str :
            Retourne une chaîne indiquant le type de fichier ('pdf', 'docx', etc.).
    """
    def detect(self, filepath: str) -> str:
        """
        Détecte le type de fichier en examinant son extension.

        Args:
            filepath (str): Le chemin vers le fichier.

        Returns:
            str: Type de fichier détecté (par exemple, 'pdf', 'docx', 'html', 'txt', ou 'unknown').
        """
        # Récupère l'extension du fichier et la convertit en minuscule
        ext = os.path.splitext(filepath)[1].lower()

        # Retourne le type de fichier en fonction de l'extension
        if ext == ".pdf":
            return "pdf"
        elif ext == ".docx":
            return "docx"
        elif ext in [".html", ".htm"]:
            return "html"
        elif ext == ".txt":
            return "txt"
        else:
            return "unknown"

def detect_file_type(filepath: str) -> str:
    """
    Fonction utilitaire pour détecter le type de fichier en utilisant TypeDetectionAgent.

    Args:
        filepath (str): Le chemin vers le fichier.

    Returns:
        str: Type de fichier détecté.
    """
    # Initialise un agent de détection et retourne le type de fichier
    agent = TypeDetectionAgent()
    return agent.detect(filepath)
