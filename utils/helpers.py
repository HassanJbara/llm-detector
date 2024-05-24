import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from ui.blocks import results_labels_html
from binoculars import Binoculars
from config import supported_detectors
from typing import List
from ghostbuster import Ghostbuster
from gptzero import GPTZero


def prepare_classifier(classifier, device=None):
    assert classifier.lower() in [x["key"].lower() for x in supported_detectors], f"Classifier {classifier} not supported!"

    if classifier.lower() == "binoculars":
        # build classifier pipeline
        return Binoculars(
            observer_name_or_path="google/gemma-2b",
            performer_name_or_path="google/gemma-1.1-2b-it"
        )
    
    if classifier.lower() == "ghostbuster":
        return Ghostbuster()

    if classifier.lower() == "gptzero":
        return GPTZero(device=device)

    # otherwise it must be simpleAI model
    # build classifier pipeline
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = [x["value"] for x in supported_detectors if x["key"].lower() == classifier.lower()][0] # this is ugly!

    classifier_pipe = pipeline(task="text-classification", model=model, device=device)

    return classifier_pipe


def process_output(classifiers: List[str], evaluations: List[float]) -> str:
    scores = [round(x * 100, 2) for x in evaluations]
    scores = [min(100, x) for x in scores] # cap at 100

    return results_labels_html(classifiers, scores)


def build_model_for_benchmark(model_name: str, quantize: bool = False, flash_attn: bool = True, device="cuda:0"):
    assert not (quantize and flash_attn), "please use either quantization or flash_attn, not both!"
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quantize else None
    dtype = torch.bfloat16 if flash_attn else None 
    attn = "flash_attention_2" if flash_attn else None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quantization_config, # do not use with flash_attn2
                                                 torch_dtype=dtype,
                                                 attn_implementation=attn,
                                                 trust_remote_code=True
                                                ).to(device)

    return model, tokenizer
