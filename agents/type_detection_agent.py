# type_detection_agent.py
import os

class TypeDetectionAgent:
    """
    Agent for detecting the file type based on the file extension.
    """
    def detect(self, filepath: str) -> str:
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

# Helper function (optional)
def detect_file_type(filepath: str) -> str:
    agent = TypeDetectionAgent()
    return agent.detect(filepath)
