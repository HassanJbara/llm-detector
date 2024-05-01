import torch
from typing import List, Dict
from transformers import pipeline
from ui_blocks import results_labels_html

supported_detectors: List[Dict[str, str]] = [
    {
        "key": "DetectGPT",
        "value": "DetectGPT"
    }, 
    {
        "key": "SimpleAI",
        "value": "Hello-SimpleAI/chatgpt-detector-roberta"
    }
]

def prepare_classifier_pipe(classifier, device=None):
    # build classifier pipeline
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = [x["value"] for x in supported_detectors if x["key"] == classifier][0] # this is ugly!

    classifier_pipe = pipeline(task="text-classification", model=model, device=device)

    return classifier_pipe

def classify(query: str, response: str, classifier: str, with_query: bool) -> str:
    classifier_pipe = prepare_classifier_pipe(classifier, device="cuda")

    sent_kwargs = {
        "truncation": True, 
        "max_length": 512, # base model context length
        # "return_all_scores": True, # deprecated 
        "top_k": None,
        # "function_to_apply": "none",
    }

    if with_query:
        evaluation = classifier_pipe(query + response, **sent_kwargs)
    else:
        evaluation = classifier_pipe(response, **sent_kwargs) 
            
    # evaluation = {classifier: [x["score"] for x in evaluation if x['label'].lower() == 'chatgpt'][0],
    #               "DetectGPT": 0.5}
    evaluation = [x["score"] for x in evaluation if x['label'].lower() == 'chatgpt'][0]
    labels = [classifier, "DetectGPT"]
    scores = list(map(lambda x: round(x * 100, 2), [evaluation, 0.5]))

    return results_labels_html(labels, scores)