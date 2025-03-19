"""
Module for translating files using multiple agents.
"""

from langchain_aws import ChatBedrock
from agents.type_detection_agent import TypeDetectionAgent
from agents.extractor_agent import ExtractorAgent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent
import os

# Initialize the language model
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    model_kwargs={"temperature": 0},
)

class MultiFileTranslator:
    def __init__(self, language_model, target_language: str = "french"):
        self.type_agent = TypeDetectionAgent()
        self.translator_agent = TranslatorAgent(language_model, target_language)

    def process_file(self, input_file: str, output_file: str):
        """
        Detects the file type for the given file, creates an ExtractorAgent
        for that type, extracts the data, translates it, then instantiates
        a GeneratorAgent (also with that type) to produce the final file.
        """
        file_type = self.type_agent.detect(input_file)
        print(f"Detected file type: {file_type}")

        # Create an ExtractorAgent using the detected file type
        extractor_agent = ExtractorAgent(file_type)
        extracted = extractor_agent.extract(input_file)
        # Our extractors return a dict with {"paragraphs": [...]}
        # If you need a list instead, do structured_data_dict["paragraphs"]

        # Translate
        translated = {}
        translated["paragraphs"] = self.translator_agent.translate(
            extracted["paragraphs"]
        )

        # Create a GeneratorAgent with the same file type
        generator_agent = GeneratorAgent(file_type)
        generator_agent.generate(translated, input_file, output_file)
        print(f"Processed {input_file} -> {output_file}")


    
if __name__ == "__main__":
    translator = MultiFileTranslator(llm, target_language="french")
    input_dir = "documents à traduire"
    output_dir = "documents traduits"
    os.makedirs(output_dir, exist_ok=True)

    input_file = "input1.pdf"
    input_filepath = os.path.join(input_dir, input_file)
    file_root, file_ext = os.path.splitext(input_file)
    output_filepath = os.path.join(output_dir, f"{file_root}_translated{file_ext}")

    translator.process_file(input_filepath, output_filepath)

    '''
    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        input_filepath = os.path.join(input_directory, filename)
        output_filepath = os.path.join(output_directory, filename)
        translator.process_file(input_filepath, output_filepath)'''
