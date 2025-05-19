"""
Module démontrant un pipeline de traduction basé sur LangGraph avec évaluation.
Ce pipeline utilise LangChain et StateGraph et ajoute un nœud d'évaluation à la fin.
"""
import cvs 
import os
import logging
from typing import TypedDict
from comet.models import download_model
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END

from agents.type_detection_agent import TypeDetectionAgent
from agents.extractor_agent import ExtractorAgent
from agents.translator_agent import TranslatorAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluation_agent import EvaluatorAgent

import csv

# Initialisation du modèle LLM pour la traduction
llm = ChatBedrock(
    client=None,
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    region_name="us-west-2",
    model_kwargs={"temperature": 0}
)
logging.getLogger("langchain_aws").setLevel(logging.ERROR)

def langgraph_pipeline(src_filepath: str, mt_filepath: str, ref_filepath: str, target_language="french", progress_callback=None, use_glossary=True, evaluate=True):
    """
    Exécute le pipeline de traduction en utilisant LangGraph.

    Ce pipeline inclut les étapes suivantes :
      - Détection du type de fichier
      - Extraction du contenu
      - Traduction
      - Génération du fichier de sortie
      - Évaluation optionnelle de la qualité de la traduction

    Args:
        src_filepath (str): Chemin vers le fichier source.
        mt_filepath (str): Chemin où le fichier traduit sera sauvegardé.
        ref_filepath (str): Chemin vers le fichier de référence pour l'évaluation.
        target_language (str, optionnel): Langue cible pour la traduction. Par défaut : 'french'.
        progress_callback (callable, optionnel): Fonction pour mettre à jour la progression.
        evaluate (bool, optionnel): Indique si l'évaluation doit être effectuée. Par défaut : True.

    Returns:
        list: Résultats de l'évaluation contenant les scores COMET (si l'évaluation est activée).
    """
    print("[LANGGRAPH] Démarrage du pipeline LangGraph...")

    # Télécharge le modèle COMET pour l'évaluation (uniquement si l'évaluation est activée)
    comet_model_path = None
    if evaluate:
        comet_model_path = download_model("Unbabel/wmt22-comet-da")

    # Définition des états utilisés dans le pipeline
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

    # Nœud pour détecter le type de fichier
    def type_detection_node(state: InputState) -> OverallState:
        file_type = TypeDetectionAgent().detect(state["src_filepath"])
        return {"file_type": file_type}

    # Nœud pour extraire le contenu du fichier
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

    # Nœud pour traduire le contenu extrait
    def translate_node(state: OverallState) -> OverallState:
        translator = TranslatorAgent(llm, target_language)
        translated_paragraphs = translator.translate(
            state["extracted_content"]["paragraphs"],
            progress_callback=progress_callback,
            terminal_progress=True,
            use_glossary=use_glossary
        )
        state["translated_paragraphs"] = translated_paragraphs
        return state

    # Nœud pour générer le fichier de sortie
    def generate_node(state: OverallState) -> OverallState:
        generator = GeneratorAgent(state["file_type"])
        structured_data = {
            "paragraphs": state["translated_paragraphs"],
        }
        generator.generate(structured_data, state["src_filepath"], state["mt_filepath"])
        return state

    # Nœud pour évaluer la qualité de la traduction
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

    # Construction du graphe d'états
    print(src_filepath, mt_filepath, ref_filepath)
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_node("TypeDetectionNode", type_detection_node)
    builder.add_node("ExtractNode", extract_node)
    builder.add_node("TranslateNode", translate_node)
    builder.add_node("GenerateNode", generate_node)

    # Ajoute le nœud d'évaluation uniquement si l'évaluation est activée
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

    # Exécution du graphe
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
    print(f"[LANGGRAPH] Terminé ! Fichier généré => {final_state['mt_filepath']}")
    if evaluate:
        print("Résultats de l'évaluation :", final_state["evaluation_results"][0]["COMET Score"])
        return final_state["evaluation_results"]
    return None

# if __name__ == "__main__":
#     # Définition des répertoires d'entrée et de sortie
#     input_dir = "src_documents"
#     output_dir = "mt_outputs"
#     ref_dir = "ref_translations"
#     os.makedirs(output_dir, exist_ok=True)

#     for input_file in ["Rapport d'audit technique Vaudrimesnil + commentaire EDPR.pdf", "SQ_15830852.pdf"]:
#         src_filepath = os.path.join(input_dir, input_file)
#         file_root, file_ext = os.path.splitext(input_file)
#         ref_filepath = os.path.join(ref_dir, input_file)

#         # évaluation sans RAG
#         mt_filepath = os.path.join(output_dir, f"{file_root}_translated_noRAG{file_ext}")
#         print(langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="english", use_glossary=False, evaluate=True))
#         # évaluation avec RAG
#         mt_filepath = os.path.join(output_dir, f"{file_root}_translated_RAG{file_ext}")
#         print(langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="english", use_glossary=True, evaluate=True))


if __name__ == "__main__":
    # Définition des répertoires d'entrée et de sortie
    input_dir = "src_documents"
    output_dir = "mt_outputs"
    ref_dir = "ref_translations"
    os.makedirs(output_dir, exist_ok=True)

    # Liste des fichiers à traiter
    test_files = ["Rapport d'audit technique Vaudrimesnil + commentaire EDPR.pdf", "SQ_15830852.pdf"]

    # Liste pour stocker les résultats
    results = []

    for input_file in test_files:
        src_filepath = os.path.join(input_dir, input_file)
        file_root, file_ext = os.path.splitext(input_file)
        ref_filepath = os.path.join(ref_dir, input_file)

        # évaluation sans RAG
        mt_filepath = os.path.join(output_dir, f"{file_root}_translated_noRAG{file_ext}")
        eval_no_rag = langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="english", use_glossary=False, evaluate=True)
        comet_score_no_rag = eval_no_rag[0]["COMET Score"] if eval_no_rag else None

        # évaluation avec RAG
        mt_filepath = os.path.join(output_dir, f"{file_root}_translated_RAG{file_ext}")
        eval_rag = langgraph_pipeline(src_filepath, mt_filepath, ref_filepath, target_language="english", use_glossary=True, evaluate=True)
        comet_score_rag = eval_rag[0]["COMET Score"] if eval_rag else None

        # Ajoute les résultats à la liste
        results.append({
            "filename": input_file,
            "comet_score_no_rag": comet_score_no_rag,
            "comet_score_rag": comet_score_rag
        })

    # Écriture des résultats dans un fichier CSV
    csv_filepath = os.path.join('comet_scores', "evaluation_results.csv")
    with open(csv_filepath, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "comet_score_no_rag", "comet_score_rag"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Résultats enregistrés dans {csv_filepath}")
