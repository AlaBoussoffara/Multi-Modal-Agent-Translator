"""
Module containing the EvaluatorAgent.

The EvaluatorAgent computes COMET scores for translation quality using standardized triplets
of original, machine-translated, and reference texts.
"""

from comet import download_model, load_from_checkpoint
import pandas as pd
import os

class EvaluatorAgent:
    """
    Evaluator agent that computes COMET scores for translation quality.

    The evaluation expects a list of dictionaries (triplets) with keys: 'filename', 'src', 'mt', 'ref'.
    """
    def __init__(self, model_path, evaluation_results_path="comet_scores"):
        """
        Initialize the evaluator with the COMET model.

        Args:
            model_path (str): Path to the COMET model.
            evaluation_results_path (str, optional): Directory to save evaluation results. Defaults to "comet_scores".
        """
        self.model = load_from_checkpoint(model_path)
        self.evaluation_results_path = evaluation_results_path

    def evaluate_triplets(self, triplets):
        """
        Evaluate a list of document triplets and return COMET scores.

        Args:
            triplets (list): List of dictionaries with keys: filename, src, mt, ref.

        Returns:
            list: Evaluation results with COMET scores.
        """
        data = [{"src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triplets]
        scores = self.model.predict(data)["scores"]
        results = [{"filename": t["filename"], "COMET Score": score} for t, score in zip(triplets, scores)]
        
        # Save results to CSV
        os.makedirs(self.evaluation_results_path, exist_ok=True)
        output_file = os.path.join(self.evaluation_results_path, "evaluation_results.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Evaluation complete! Results saved in '{output_file}'")
        return results

if __name__ == "__main__":
    # Example usage: download and evaluate using the COMET model.
    model_path = download_model("Unbabel/wmt22-comet-da")
    evaluator = EvaluatorAgent(model_path)
    sample_triplets = [
        {
            "filename": "SQ_15830852.pdf",
            "src": "This is the original text.",
            "mt": "This is the translated text.",
            "ref": "This is the reference text."
        }
    ]
    evaluator.evaluate_triplets(sample_triplets)
