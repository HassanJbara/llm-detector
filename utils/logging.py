import os
import json
import statistics
from typing import List

from rich import print
from rich.table import Table


def prepare_results_table(classifier: str, n_samples: int, classifier_evaluations: List[float], classifier_labels: List[bool], texts: List[str]):
    table = Table(title=classifier, show_footer=True)

    avgs_eval = "{:.2f}".format(statistics.mean(classifier_evaluations))
    avgs_label = "{:.2f}".format((sum(classifier_labels) / n_samples) * 100)
    
    table.add_column("i",)
    table.add_column("Text", style="green")
    table.add_column("Score", f"[b]{avgs_eval}", style="cyan",)
    table.add_column("Label", f"[b]{avgs_label}", style="magenta")
    
    for i in range(len(classifier_evaluations)): # or labels
        table.add_row(str(i), f"{texts[i][:75]}...", "{:.2f}".format(classifier_evaluations[i]), str(classifier_labels[i]))
    
    return table


def output_results_json(model: str, classifiers: List[str], n_samples: int, evaluations, labels, texts: List[str]):
    directory = f"benchmarks/{model}"
    file_name = f"{n_samples}.json"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    data = {"Scores": [], "Labels": [], "avg_scores": {}, "avg_correct_labels": {}}
    
    for i in range(n_samples):
        scores_row = {"i": i, "Text": texts[i]}
        labels_row = {"i": i, "Text": texts[i]}
        
        for classifier in classifiers:
            scores_row[classifier] = evaluations[classifier][i]
            labels_row[classifier] = bool(labels[classifier][i]) # to avoid a bool_ json bug
                
        data["Scores"].append(scores_row)
        data["Labels"].append(labels_row)

    # add averages
    for classifier in classifiers: 
        avg_eval = statistics.mean(evaluations[classifier])
        avg_label = (sum(labels[classifier]) / n_samples) * 100
        
        data["avg_scores"][classifier] = avg_eval
        data["avg_correct_labels"][classifier] = avg_label
    
    with open(f"{directory}/{file_name}", 'w') as file:
        json.dump(data, file, indent=4)

        
        

    