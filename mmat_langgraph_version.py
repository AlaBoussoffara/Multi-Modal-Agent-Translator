"""
Module demonstrating a LangGraph-based translator pipeline with evaluation.
This pipeline leverages LangChain and StateGraph and adds an evaluation node at the end.
"""

import os
import logging
from typing import TypedDict
from comet import download_model
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

from agents.type_detection_agent import TypeDetectionAgent
from agents.extractor_agent import ExtractorAgent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluation_agent import EvaluatorAgent

# Initialize the LLM model for translation
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0}
)
logging.getLogger("langchain_aws").setLevel(logging.ERROR)

def langgraph_pipeline(src_filepath: str, mt_filepath: str, ref_filepath: str, target_language="french", progress_callback=None, evaluate=True):
    """
    Execute the translation pipeline using LangGraph.

    This function defines a state graph that includes:
      - Type detection
      - Content extraction
      - Translation
      - Generation of the output file
      - Optional evaluation of the translation quality

    Args:
        src_filepath (str): Path to the input file.
        mt_filepath (str): Path where the translated file will be saved.
        ref_filepath (str): Path to the reference file for evaluation.
        target_language (str, optional): Target language for translation. Default is 'french'.
        progress_callback (callable, optional): Function to update progress status.
        evaluate (bool, optional): Whether to perform evaluation. Default is True.

    Returns:
        list: Evaluation results containing COMET scores (if evaluation is enabled).
    """
    print("[LANGGRAPH] Starting LangGraph pipeline...")

    # Download the COMET model for evaluation (only if evaluation is enabled)
    comet_model_path = None
    if evaluate:
        comet_model_path = download_model("Unbabel/wmt22-comet-da")

    class OverallState(TypedDict):
        src_filepath: str
        mt_filepath: str
        ref_filepath: str
        file_type: str
        extracted_content: dict
        ref_content: dict
        translated_paragraphs: list
        evaluation_results: list

    class InputState(TypedDict):
        src_filepath: str
        mt_filepath: str
        ref_filepath: str

    class OutputState(TypedDict):
        mt_filepath: str
        evaluation_results: list

    def type_detection_node(state: InputState) -> OverallState:
        file_type = TypeDetectionAgent().detect(state["src_filepath"])
        return {"file_type": file_type}

    def extract_node(state: OverallState) -> OverallState:
        extractor = ExtractorAgent(state["file_type"])
        original_content = extractor.extract(state["src_filepath"])
        state["extracted_content"] = original_content

        # Lire le fichier de référence uniquement si l'évaluation est activée
        if evaluate:
            ref_content = extractor.extract(state["ref_filepath"])
            state["ref_content"] = ref_content
        else:
            state["ref_content"] = None  # Pas de contenu de référence si l'évaluation est désactivée

        return state

    def translate_node(state: OverallState) -> OverallState:
        translator = TranslatorAgent(llm, target_language)
        translated_paragraphs = translator.translate(
            state["extracted_content"]["paragraphs"],
            progress_callback=progress_callback,
            terminal_progress=True
        )
        state["translated_paragraphs"] = translated_paragraphs
        return state

    def generate_node(state: OverallState) -> OverallState:
        generator = GeneratorAgent(state["file_type"])
        structured_data = {"paragraphs": state["translated_paragraphs"]}
        generator.generate(structured_data, state["src_filepath"], state["mt_filepath"])
        return state

    def evaluate_node(state: OverallState) -> OutputState:
        filename = os.path.basename(state["src_filepath"])
        src_text = " ".join([para["text"] for para in state["extracted_content"].get("paragraphs", [])])
        mt_text = " ".join([para["text"] for para in state["translated_paragraphs"]])
        ref_text = " ".join([para["text"] for para in state["ref_content"].get("paragraphs", [])])
        triplet = {"filename": filename, "src": src_text, "mt": mt_text, "ref": ref_text}
        
        evaluator = EvaluatorAgent(comet_model_path)
        evaluation_results = evaluator.evaluate_triplets([triplet])
        state["evaluation_results"] = evaluation_results
        return {"mt_filepath": state["mt_filepath"], "evaluation_results": evaluation_results}

    print(src_filepath, mt_filepath, ref_filepath)
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_node("TypeDetectionNode", type_detection_node)
    builder.add_node("ExtractNode", extract_node)
    builder.add_node("TranslateNode", translate_node)
    builder.add_node("GenerateNode", generate_node)

    # Add the evaluation node only if evaluation is enabled
    if evaluate:
        builder.add_node("EvaluateNode", evaluate_node)
        builder.add_edge("GenerateNode", "EvaluateNode")
        builder.add_edge("EvaluateNode", END)
    else:
        builder.add_edge("GenerateNode", END)

    builder.add_edge(START, "TypeDetectionNode")
    builder.add_edge("TypeDetectionNode", "ExtractNode")
    builder.add_edge("ExtractNode", "TranslateNode")
    builder.add_edge("TranslateNode", "GenerateNode")

    graph = builder.compile()
    print({
        "src_filepath": src_filepath,
        "mt_filepath": mt_filepath,
        "ref_filepath": ref_filepath
    })
    final_state = graph.invoke({
        "src_filepath": src_filepath,
        "mt_filepath": mt_filepath,
        "ref_filepath": ref_filepath
    })
    print(f"[LANGGRAPH] Done! Output => {final_state['mt_filepath']}")
    if evaluate:
        print("Evaluation Results:", final_state["evaluation_results"][0]["COMET Score"])
        return final_state["evaluation_results"]
    return None

if __name__ == "__main__":
    input_dir = "src_documents"
    output_dir = "mt_outputs"
    ref_dir = "ref_translations"
    os.makedirs(output_dir, exist_ok=True)

    input_file = "Consigne_preetude_poleIA_1A.pdf"
    src_filepath = os.path.join(input_dir, input_file)
    file_root, file_ext = os.path.splitext(input_file)
    mt_filepath = os.path.join(output_dir, f"{file_root}_translated{file_ext}")
    ref_filepath = os.path.join(ref_dir, input_file)

    # Example: Run the pipeline with evaluation enabled
    # print(langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="french", evaluate=True))

    # Example: Run the pipeline without evaluation
    print(langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="french", evaluate=False))
