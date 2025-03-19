import os
from pdf2docx import Converter  # Convert PDF to DOCX
from docx2pdf import convert  # Convert DOCX back to PDF
from typing import Optional
from word2word import WordTranslator  # Import the word-to-word translator

class PDFTranslator:
    """
    Converts a PDF to a structured format, translates it using `WordTranslator`, 
    and generates a translated PDF while preserving layout.
    """

    def __init__(self, target_language="french"):
        self.target_language = target_language

    def pdf_to_docx(self, pdf_path: str, docx_path: str):
        """ Converts a PDF file to a DOCX file while preserving layout. """
        print("Converting PDF to DOCX...")
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        print("✅ PDF converted to DOCX")

    def translate_docx(self, input_docx: str, output_docx: str):
        """ Uses `WordTranslator` to translate the DOCX file. """
        print("Translating DOCX using WordTranslator...")
        word_translator = WordTranslator(input_docx, output_docx, self.target_language)
        word_translator.translate_document()
        print("✅ DOCX translation completed")

    def docx_to_pdf(self, docx_path: str, pdf_path: str):
        """ Converts a translated DOCX back to PDF while keeping formatting. """
        print("Converting translated DOCX to PDF...")
        convert(docx_path, pdf_path)
        print(f"✅ Translated PDF saved at: {pdf_path}")

    def translate_pdf(self, pdf_path: str, output_pdf: Optional[str] = None):
        """ Converts a PDF to a translated PDF via DOCX. """
        if not output_pdf:
            output_pdf = pdf_path.replace(".pdf", "_translated.pdf")

        docx_path = pdf_path.replace(".pdf", ".docx")
        translated_docx_path = docx_path.replace(".docx", "_translated.docx")

        # Step 1: Convert PDF to DOCX
        self.pdf_to_docx(pdf_path, docx_path)

        # Step 2: Translate DOCX using WordTranslator
        self.translate_docx(docx_path, translated_docx_path)

        # Step 3: Convert translated DOCX to PDF
        self.docx_to_pdf(translated_docx_path, output_pdf)

        # Optional: Clean up intermediate DOCX files
        os.remove(docx_path)
        os.remove(translated_docx_path)

        return output_pdf


# Example Usage
if __name__ == "__main__":
    # Initialize PDF translator with target language
    pdf_translator = PDFTranslator(target_language="english")

    # Translate PDF
    pdf_translator.translate_pdf("Rapport d'audit technique Vaudrimesnil + commentaire EDPR.pdf")
