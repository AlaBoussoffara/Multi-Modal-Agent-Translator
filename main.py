"""
main.py

Demonstrates 2 approaches to a multi-step translator pipeline:
1) A "Classic" direct code-driven pipeline (no LLM agent).
2) A LangGraph approach.

"""

import os
from typing import TypedDict
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

# Import agents
from type_detection_agent import TypeDetectionAgent
from extractor_agent import ExtractorAgent
from translator_agent import TranslatorAgent
from generator_agent import GeneratorAgent

# Initialize the language model
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0},
)

##############################
# 1) CLASSIC PIPELINE
##############################
def classic_pipeline(input_filepath: str, output_filepath: str):
    """
    A direct, code-driven pipeline (no LLM deciding the chain).
    1) Detect file type
    2) Extract content
    3) Translate content
    4) Generate final output
    """
    print("[CLASSIC] Starting direct pipeline...")

    # 1) Detect file type
    file_type = TypeDetectionAgent().detect(input_filepath)

    # 2) Extract content
    extractor = ExtractorAgent(file_type)
    extracted_content = extractor.extract(input_filepath)  # e.g. {"paragraphs": [...]}

    paragraphs = extracted_content["paragraphs"]

    # 3) Translate content
    translator = TranslatorAgent(llm)
    translated_paragraphs = translator.translate(paragraphs)  # returns updated list or dict

    # 4) Generate final output
    generator = GeneratorAgent(file_type)
    structured_data = {"paragraphs": translated_paragraphs}
    generator.generate(structured_data, input_filepath, output_filepath)

    print(f"[CLASSIC] Done! Output => {output_filepath}")


##############################
# 2) LANGGRAPH PIPELINE
##############################
def langgraph_pipeline(input_filepath: str, output_filepath: str):
    """
    A LangGraph-based pipeline.
    """
    print("[LANGGRAPH] Starting LangGraph pipeline...")

    # Define state types
    class OverallState(TypedDict):
        input_filepath: str
        output_filepath: str
        file_type: str
        extracted_content: dict
        translated_paragraphs: list

    class InputState(TypedDict):
        input_filepath: str
        output_filepath: str

    class OutputState(TypedDict):
        output_filepath: str

    # Define pipeline nodes
    def type_detection_node(state: InputState) -> OverallState:
        file_type = TypeDetectionAgent().detect(state["input_filepath"])
        return {"file_type": file_type}

    def extract_node(state: OverallState) -> OverallState:
        extractor = ExtractorAgent(state["file_type"])
        extracted_content = extractor.extract(state["input_filepath"])
        return {"extracted_content": extracted_content}

    def translate_node(state: OverallState) -> OverallState:
        translator = TranslatorAgent(llm)
        translated_paragraphs = translator.translate(state["extracted_content"]["paragraphs"])
        return {"translated_paragraphs": translated_paragraphs}

    def generate_node(state: OverallState) -> OutputState:
        generator = GeneratorAgent(state["file_type"])
        structured_data = {"paragraphs": state["translated_paragraphs"]}
        generator.generate(structured_data, state["input_filepath"], state["output_filepath"])
        return {"output_filepath": state["output_filepath"]}

    # Build the state graph
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_node("TypeDetectionNode", type_detection_node)
    builder.add_node("ExtractNode", extract_node)
    builder.add_node("TranslateNode", translate_node)
    builder.add_node("GenerateNode", generate_node)
    builder.add_edge(START, "TypeDetectionNode")
    builder.add_edge("TypeDetectionNode", "ExtractNode")
    builder.add_edge("ExtractNode", "TranslateNode")
    builder.add_edge("TranslateNode", "GenerateNode")
    builder.add_edge("GenerateNode", END)

    # Compile and invoke the graph
    graph = builder.compile()
    graph.invoke({"input_filepath": input_filepath, "output_filepath": output_filepath})

    print(f"[LANGGRAPH] Done! Output => {output_filepath}")


##############################
# MAIN
##############################
if __name__ == "__main__":
    input_dir = "documents_to_translate"
    output_dir = "translated_documents"
    os.makedirs(output_dir, exist_ok=True)

    input_file = "input.pdf"
    input_filepath = os.path.join(input_dir, input_file)
    file_root, file_ext = os.path.splitext(input_file)
    output_filepath = os.path.join(output_dir, f"{file_root}_translated{file_ext}")

    print("\n=== APPROACH 1: CLASSIC PIPELINE ===")
    classic_pipeline(input_filepath, output_filepath)

    print("\n=== APPROACH 2: LANGGRAPH PIPELINE ===")
    langgraph_pipeline(input_filepath, output_filepath)
