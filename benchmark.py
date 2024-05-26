import json
import random
import argparse
import statistics
from pathlib import Path
from typing import List

from utils.classify import classify_with_simpleai, classify_with_binoculars, classify_with_ghostbuster, classify_with_gptzero
from utils.helpers import build_model_for_benchmark, prepare_classifier
from utils.logging import prepare_results_table, output_results_json
from utils.dataset import build_dataset
from config import supported_detectors
from tqdm import tqdm

from rich import print
from rich.console import Console
from rich.style import Style


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="hassanjbara/LONG", help="dataset name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="model name")
    parser.add_argument("--classifiers", type=str, default="simpleai", help="classifiers names separated by a comma")
    
    parser.add_argument("--samples", type=int, default=100, help="number of samples to test the model on")
    parser.add_argument("--with_query", type=bool, default=False, help="whether to include the query during evaluation")
    
    parser.add_argument("--quantize", type=bool, default=False, help="whether to load model in 8bit or not")
    parser.add_argument("--flash_attn", type=bool, default=True, help="whether to use flash_attn 2 or not") 
    # parser.add_argument("--use_peft", type=bool, default=False, help="whether to use peft or not")
    
    parser.add_argument("--model_device", type=str, default="cuda:0", help="which device to load the model to")
    parser.add_argument("--classifiers_device", type=str, default="cuda:0", help="which device to load the classifiers to")

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def main(args):
    assert not (args.quantize and args.flash_attn), "You can use both flash_attn and quantization at the same time!"

    console = Console()
    logging_style = Style(color="green", bold=True)
    
    classifiers = args.classifiers.split(',')
    for classifier in classifiers:
        assert classifier in [x["key"].lower() for x in supported_detectors], f"Classifier {classifier} not supported!"
    use_sys_prompt = True if "llama" in args.model_name.lower() else False
    
    classifier_models = {classifier: prepare_classifier(classifier, args.classifiers_device) for classifier in classifiers}
    model, tokenizer = build_model_for_benchmark(args.model_name, args.quantize, args.flash_attn, args.model_device)
    dataset = build_dataset(tokenizer, args.dataset, sys_prompt=use_sys_prompt, padding=False)
    
    # use with llama-3
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    texts = []
    evaluations = {classifier: [] for classifier in classifiers}
    labels = {classifier: [] for classifier in classifiers}
    
    console.print("Generate texts", style=logging_style)
    for i in tqdm(range(args.samples)):
        sample_index = random.randint(0, len(dataset)-1)
        outputs = model.generate(dataset[sample_index]['input_ids'].unsqueeze(dim=0).to(args.model_device), 
                                 attention_mask=dataset[sample_index]['attention_mask'].unsqueeze(dim=0).to(args.model_device), 
                                 max_new_tokens=512, 
                                 eos_token_id=terminators,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        q_len = len(tokenizer.decode(dataset[sample_index]['input_ids'], skip_special_tokens=True))

        text = text if args.with_query else text[q_len:]
        texts.append(text)

    for classifier in evaluations:
        console.print(f"Classifying with {classifier}", style=logging_style)
        classifier_model = classifier_models[classifier]
        
        for i in tqdm(range(args.samples)):
            text = texts[i]
            
            if classifier.lower()=="binoculars":
                evaluation, label = classify_with_binoculars(text, args.classifiers_device, classifier_model, return_label=True)
            if classifier.lower()=="simpleai":
                evaluation, label = classify_with_simpleai(text, args.classifiers_device, classifier_model, return_label=True)
            if classifier.lower()=="ghostbuster":
                evaluation, label = classify_with_ghostbuster(text, args.classifiers_device, classifier_model, return_label=True)
            if classifier.lower()=="gptzero":
                evaluation, label = classify_with_gptzero(text, args.classifiers_device, classifier_model, return_label=True)

            evaluations[classifier].append(evaluation)
            labels[classifier].append(label)

    for classifier in classifiers:
        table = prepare_results_table(classifier, args.samples, evaluations[classifier], labels[classifier], texts)
        console.print(table)
    
    output_results_json(args.model_name, classifiers, args.samples, evaluations, labels, texts)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)