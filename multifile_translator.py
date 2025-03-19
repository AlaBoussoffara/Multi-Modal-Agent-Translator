# multifile_translator.py

from type_detection_agent import TypeDetectionAgent
from extractor_agent import ExtractorAgent
from translator_agent import TranslatorAgent
from generator_agent import GeneratorAgent

class MultiFileTranslator:
    """
    Orchestrates the translation process using four agents:
      1. TypeDetectionAgent: Determines the file type.
      2. ExtractorAgent: Extracts structured content (initialized with file_type).
      3. TranslatorAgent: Translates the content.
      4. GeneratorAgent: Generates the output file (initialized with file_type).
    """
    def __init__(self, llm, target_language: str = "french"):
        # The TypeDetectionAgent is still used to figure out file type,
        # but we no longer store a single ExtractorAgent here because we must
        # instantiate one with the file type for each file we process.
        self.type_agent = TypeDetectionAgent()
        self.translator_agent = TranslatorAgent(llm, target_language)

        # The extractor and generator agents will be created per-file below
        # because each must be initialized with the correct file_type.

    def process_file(self, input_filepath: str, output_filepath: str):
        """
        Detects the file type for the given file, creates an ExtractorAgent
        for that type, extracts the data, translates it, then instantiates
        a GeneratorAgent (also with that type) to produce the final file.
        """
        file_type = self.type_agent.detect(input_filepath)
        print(f"Detected file type: {file_type}")

        # Create an ExtractorAgent using the detected file type
        extractor_agent = ExtractorAgent(file_type)
        structured_data_dict = extractor_agent.extract(input_filepath)
        # Our extractors return a dict with {"paragraphs": [...]}
        # If you need a list instead, do structured_data_dict["paragraphs"]

        # Translate
        translated_data_dict = {}
        translated_data_dict["paragraphs"] = self.translator_agent.translate(
            structured_data_dict["paragraphs"]
        )

        # Create a GeneratorAgent with the same file type
        generator_agent = GeneratorAgent(file_type)
        generator_agent.generate(translated_data_dict, input_filepath, output_filepath)
        print(f"Processed {input_filepath} -> {output_filepath}")
