from comet import download_model, load_from_checkpoint
from docx import Document
import pdfplumber

class EvaluatorAgent(): 
    """This class is responsible for evaluating the performance of the translation agent. 
    It uses the COMET framework to give a score to the translation"""
    
    def __init__(self, model_path):
        self.model = load_from_checkpoint(model_path)