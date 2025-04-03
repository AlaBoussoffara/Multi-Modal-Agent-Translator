"""
Module demonstrating a classic translator pipeline with evaluation.
This version calls the agents sequentially without using LangChain or StateGraph.
"""

import os
import logging
from comet import download_model
from agents.type_detection_agent import TypeDetectionAgent
from agents.extractor_agent import ExtractorAgent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluation_agent import EvaluatorAgent
from langchain_aws import ChatBedrock

# Initialize the LLM model for translation
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0},
)
logging.getLogger("langchain_aws").setLevel(logging.ERROR)

def classic_pipeline(input_filepath: str, output_filepath: str, ref_filepath: str, progress_callback=None):
    """
    Run the classic translation pipeline: detect file type, extract content, translate, generate output, and evaluate.

    Args:
        input_filepath (str): Path to the input file.
        output_filepath (str): Path where the translated file will be saved.
        ref_filepath (str): Path to the reference translation for evaluation.
        progress_callback (callable, optional): Function to update progress status.
    """
    print("[CLASSIC PIPELINE] Starting classic pipeline...")

    # 1. Detect file type
    file_type = TypeDetectionAgent().detect(input_filepath)
    print(f"Detected file type: {file_type}")

    # 2. Extract original and reference content
    extractor = ExtractorAgent(file_type)
    original_content = extractor.extract(input_filepath)
    ref_content = extractor.extract(ref_filepath)

    # 3. Translate content
    translator = TranslatorAgent(model=llm, target_language="french", max_chunk_words=20)
    translated_paragraphs = translator.translate(
        original_content["paragraphs"],
        progress_callback=progress_callback,
        terminal_progress=True
    )

    # 4. Generate translated output file
    generator = GeneratorAgent(file_type)
    structured_data = {"paragraphs": translated_paragraphs}
    generator.generate(structured_data, input_filepath, output_filepath)
    print(f"Generated translated document: {output_filepath}")

    # 5. Evaluate the translation using the standardized triplet
    filename = os.path.basename(input_filepath)
    src_text = " ".join([para["text"] for para in original_content.get("paragraphs", [])])
    mt_text = " ".join([para["text"] for para in translated_paragraphs])
    ref_text = " ".join([para["text"] for para in ref_content.get("paragraphs", [])])
    triplet = {"filename": filename, "src": src_text, "mt": mt_text, "ref": ref_text}

    # Download and use the COMET model for evaluation
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    evaluator = EvaluatorAgent(comet_model_path)
    evaluation_results = evaluator.evaluate_triplets([triplet])
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    # Updated directory names following COMET vocabulary
    input_dir = "src_documents"
    output_dir = "mt_outputs"
    ref_dir = "ref_translations"
    os.makedirs(output_dir, exist_ok=True)

    input_file = "SQ_15830852.pdf"
    input_filepath = os.path.join(input_dir, input_file)
    file_root, file_ext = os.path.splitext(input_file)
    output_filepath = os.path.join(output_dir, f"{file_root}_translated{file_ext}")
    ref_filepath = os.path.join(ref_dir, input_file)

    classic_pipeline(input_filepath, output_filepath, ref_filepath)
