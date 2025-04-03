"""
This module provides functionality to detect the file type based on its extension.
Classes:
    TypeDetectionAgent: A class with a method to detect file types.
Functions:
    detect_file_type(filepath: str) -> str: A helper function to detect file types.
"""
import os

class TypeDetectionAgent:
    """
    Agent for detecting the file type based on its extension.

    Methods:
        detect(filepath: str) -> str:
            Returns a string indicating the file type ('pdf', 'docx', etc.).
    """
    def detect(self, filepath: str) -> str:
        """
        Detect the file type by examining the file extension.

        Args:
            filepath (str): The path to the file.

        Returns:
            str: Detected file type (e.g., 'pdf', 'docx', 'html', 'txt', or 'unknown').
        """
        ext = os.path.splitext(filepath)[1].lower()
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
    Helper function to detect file type using TypeDetectionAgent.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: Detected file type.
    """
    agent = TypeDetectionAgent()
    return agent.detect(filepath)
