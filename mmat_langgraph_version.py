"""
main.py

Demonstrates a LangGraph approach to a multi-step translator pipeline.

"""

import os
from typing import TypedDict
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

# Import agents
from agents.type_detection_agent import TypeDetectionAgent
from agents.extractor_agent import ExtractorAgent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent

# Initialize the language model
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0},
)

def langgraph_pipeline(input_filepath: str, output_filepath: str, target_language="french", progress_callback=None):
    """
    A LangGraph-based pipeline with progress tracking for continuous updates.
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
        translator = TranslatorAgent(llm, target_language)
        
        # Track progress
        total_chunks = len(state["extracted_content"]["paragraphs"])+1 
        completed_chunks = 0

        # Translate paragraphs and update progress
        translated_paragraphs = []
        for paragraph in state["extracted_content"]["paragraphs"]:
            translated_paragraph = translator.translate([paragraph])[0]
            translated_paragraphs.append(translated_paragraph)
            completed_chunks += 1

            # Update the progress bar
            if progress_callback:
                progress_percentage = int((completed_chunks / total_chunks) * 100)
                progress_callback(progress_percentage)

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
    input_dir = "documents à traduire"
    output_dir = "documents traduits"
    os.makedirs(output_dir, exist_ok=True)

    input_file = "input1.pdf"
    input_filepath = os.path.join(input_dir, input_file)
    file_root, file_ext = os.path.splitext(input_file)
    output_filepath = os.path.join(output_dir, f"{file_root}_translated{file_ext}")

    print("\n=== LANGGRAPH PIPELINE ===")
    langgraph_pipeline(input_filepath, output_filepath)
