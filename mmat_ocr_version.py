"""
main_ocr.py

LangGraph pipeline adapté à l'OCR avec PyTesseract.
"""

import os
from typing import TypedDict
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

# Import agents
from agents.ocr_agent_pytesseract import OCR_Agent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent

# Initialise le modèle de langue (Llama3 sur Bedrock ici)
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0},
)

def langgraph_ocr_pipeline(input_filepath: str, output_filepath: str, target_language="french", progress_callback=None):
    """
    Pipeline LangGraph pour OCR + traduction + génération.
    """
    print("[LANGGRAPH] Starting OCR LangGraph pipeline...")

    # Définition des types d'état
    class OverallState(TypedDict):
        input_filepath: str
        output_filepath: str
        extracted_content: dict
        translated_paragraphs: list

    class InputState(TypedDict):
        input_filepath: str
        output_filepath: str

    class OutputState(TypedDict):
        output_filepath: str

    # Nœud d'extraction OCR
    def extract_node(state: OverallState) -> OverallState:
        ocr_agent = OCR_Agent("Multi-Modal-Agent-Translator/documents_a_traduire/SQ_15830852.PDF")
        extracted_content = ocr_agent.extract_blocks(state["input_filepath"])
        
        # Extraction des textes simples pour la traduction
        paragraphs = [block["text"] for page in extracted_content["pages"] for block in page["blocks"]]
        return {
            "extracted_content": {
                "paragraphs": paragraphs,
                "structured": extracted_content
            }
        }

    # Nœud de traduction
    def translate_node(state: OverallState) -> OverallState:
        translator = TranslatorAgent(llm, target_language)
        
        total_chunks = len(state["extracted_content"]["paragraphs"])
        completed_chunks = 0
        translated_paragraphs = []

        for paragraph in state["extracted_content"]["paragraphs"]:
            translated = translator.translate([paragraph])[0]
            translated_paragraphs.append(translated)
            completed_chunks += 1

            if progress_callback:
                progress_percentage = int((completed_chunks / total_chunks) * 100)
                progress_callback(progress_percentage)

        return {"translated_paragraphs": translated_paragraphs}

    # Nœud de génération du document final
    def generate_node(state: OverallState) -> OutputState:
        generator = GeneratorAgent("ocr")  # "ocr" pour indiquer une version spécialisée ?
        structured_data = {
            "paragraphs": state["translated_paragraphs"],
            "structured": state["extracted_content"]["structured"]
        }
        generator.generate(structured_data, state["input_filepath"], state["output_filepath"])
        return {"output_filepath": state["output_filepath"]}

    # Construction du graphe
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_node("ExtractNode", extract_node)
    builder.add_node("TranslateNode", translate_node)
    builder.add_node("GenerateNode", generate_node)
    builder.add_edge(START, "ExtractNode")
    builder.add_edge("ExtractNode", "TranslateNode")
    builder.add_edge("TranslateNode", "GenerateNode")
    builder.add_edge("GenerateNode", END)

    # Compilation et exécution
    graph = builder.compile()
    graph.invoke({
        "input_filepath": input_filepath,
        "output_filepath": output_filepath
    })

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

    print("\n=== LANGGRAPH OCR PIPELINE ===")
    langgraph_ocr_pipeline(input_filepath, output_filepath)

