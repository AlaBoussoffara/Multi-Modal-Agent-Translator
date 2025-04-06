"""
This module contains the EvaluationAgent. 
It is a class for evaluating the performance of the translation agent.
It uses the COMET framework to score the translations. 
The score is based on the original text and reference translations."""
import os
from comet import download_model, load_from_checkpoint
from docx import Document
import pdfplumber
import pandas as pd

class EvaluatorAgent():
    """This class is responsible for evaluating the performance of the translation agent. 
    It uses the COMET framework to give a score to the translation"""
    def __init__(self, model_path, original_files_path, translated_files_path, ref_files_path, evaluation_results_path="evaluation_results"):
        self.model = load_from_checkpoint(model_path)
        self.original_files_path = original_files_path
        self.translated_files_path = translated_files_path
        self.ref_files_path = ref_files_path
        self.evaluation_results_path = evaluation_results_path
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    def extract_text_from_docx(self, docx_path):
        """Extract text from a DOCX file."""
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    def prepare_document_triplets(self):
        """Prepare triplets of original, translated, and reference documents for evaluation."""
        # Assuming the original and translated documents are in the same folder structure
        translated_folder = self.translated_files_path
        original_folder = self.original_files_path
        reference_folder = self.ref_files_path
        file_triplets = []
        for filename in os.listdir(original_folder):
            if filename in os.listdir(translated_folder) and filename in os.listdir(reference_folder):
                print("FILENAME:", filename)
                if filename.endswith((".pdf", ".PDF", ".docx")):
                    original_path = os.path.join(original_folder, filename)
                    translated_path = os.path.join(translated_folder, filename)
                    reference_path = os.path.join(reference_folder, filename)
                    if filename.endswith((".pdf", ".PDF")):
                        source_text = self.extract_text_from_pdf(original_path)
                        translated_text = self.extract_text_from_pdf(translated_path)
                        reference_text = self.extract_text_from_pdf(reference_path)
                    else:
                        source_text = self.extract_text_from_docx(original_path)
                        translated_text = self.extract_text_from_docx(translated_path)
                        reference_text = self.extract_text_from_docx(reference_path)

                    file_triplets.append({"filename": filename, "src": source_text, "mt": translated_text, "ref": reference_text})

        return file_triplets
    def evaluate_documents(self):
        """ Evaluate the translated documents using the COMET model."""
        file_triplets = self.prepare_document_triplets()
        data = [{"src": triplet["src"], "mt": triplet["mt"], "ref": triplet["ref"]} for triplet in file_triplets]
        # Batch Processing in COMET
        scores = self.model.predict(data)["scores"]  # Vectorized batch processing
        # Store results
        results = [{"Filename": triplet["filename"], "COMET Score": score} for triplet, score in zip(file_triplets, scores)]
        df = pd.DataFrame(results)
        output_folder = self.evaluation_results_path
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
        output_file = os.path.join(self.evaluation_results_path, "evaluation_results.csv")
        df.to_csv(output_file, index=False)
        print("Processing complete! Results saved in 'evaluation_results.csv'.")

    def evaluate_without_reference(self, original_file_path, translated_file_path):
        """
        Evaluate a single pair of original and translated files without a reference.
        """
        # Extract text from the original file
        if original_file_path.endswith((".pdf", ".PDF")):
            source_text = self.extract_text_from_pdf(original_file_path)
        elif original_file_path.endswith(".docx"):
            source_text = self.extract_text_from_docx(original_file_path)
        else:
            raise ValueError("Unsupported file format for the original file.")

        # Extract text from the translated file
        if translated_file_path.endswith((".pdf", ".PDF")):
            translated_text = self.extract_text_from_pdf(translated_file_path)
        elif translated_file_path.endswith(".docx"):
            translated_text = self.extract_text_from_docx(translated_file_path)
        else:
            raise ValueError("Unsupported file format for the translated file.")

        # Prepare data for COMET
        data = [{"src": source_text, "mt": translated_text}]
        print(data)

        # Get the COMET score
        score = self.model.predict(data)["scores"][0]

        print(f"COMET Score (without reference): {score}")
        return score

if __name__ == "__main__":
    # Example usage
    # model_path = download_model("Unbabel/wmt22-comet-da")  # Path to your COMET model
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    original_files_path = "documents_a_traduire"
    translated_files_path = "documents_traduits"
    ref_files_path = "references"
    evaluator = EvaluatorAgent(model_path, original_files_path, translated_files_path, ref_files_path)
    
    # Evaluate all documents with references
    # evaluator.evaluate_documents()

    # Test evaluation without reference for two specific files
    # original_file = "documents_a_traduire/SQ_13124846.pdf"  # Replace with the actual path to your original file
    # translated_file = "documents_traduits/SQ_15830852.pdf"  # Replace with the actual path to your translated file
    # evaluator.evaluate_without_reference(original_file, translated_file)