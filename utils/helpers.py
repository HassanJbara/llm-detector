import torch
from transformers import pipeline
from ui.blocks import results_labels_html
from binoculars import Binoculars
from config import supported_detectors
from typing import List
from ghostbuster import Ghostbuster

def prepare_classifier(classifier, device=None):
    assert classifier in [x["key"] for x in supported_detectors], f"Classifier {classifier} not supported!"

    if classifier.lower() == "binoculars":
        # build classifier pipeline
        return Binoculars(
            observer_name_or_path="google/gemma-2b",
            performer_name_or_path="google/gemma-1.1-2b-it"
        )
    
    if classifier.lower() == "ghostbuster":
        return Ghostbuster()

    # otherwise it must be simpleAI model
    # build classifier pipeline
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = [x["value"] for x in supported_detectors if x["key"] == classifier][0] # this is ugly!

    classifier_pipe = pipeline(task="text-classification", model=model, device=device)

    return classifier_pipe


def process_output(classifiers: List[str], evaluations: List[float]) -> str:
    scores = [round(x * 100, 2) for x in evaluations]
    scores = [min(100, x) for x in scores] # cap at 100

    return results_labels_html(classifiers, scores)


