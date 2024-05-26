from typing import List, Union
from utils.helpers import prepare_classifier, process_output


def classify_with_simpleai(text: str, device: str, simpleai=None, return_label: bool = False) -> Union[float, bool]:
    if simpleai==None:
        classifier = "SimpleAI"
        simpleai = prepare_classifier(classifier, device=device)

    sent_kwargs = {
        "truncation": True, 
        "max_length": 512, # base model context length
        "top_k": None,
    }

    evaluation = simpleai(text, **sent_kwargs) # returns a list of dictionaries
    evaluation = [x["score"] for x in evaluation if x['label'].lower() == 'chatgpt'][0]  
    
    if return_label:
        label = evaluation > 0.5
        return evaluation, label

    return evaluation


def classify_with_binoculars(text: str, device: str, bino=None, return_label: bool = False) -> Union[float, bool]:
    if bino==None:
        classifier = "Binoculars"
        bino = prepare_classifier(classifier=classifier, device=device)

    evaluation = bino.compute_score(text)
    
    if return_label:
        label = "ai" in bino.predict(text).lower() # returns either "Most likely AI-generated" or "Most likely human-generated"
        return evaluation, label

    return evaluation


def classify_with_ghostbuster(text: str, device: str, ghostbuster=None, return_label: bool = False) -> Union[float, bool]:
    if ghostbuster==None:
        classifier = "Ghostbuster"
        ghostbuster = prepare_classifier(classifier=classifier, device=device)
    
    evaluation = ghostbuster.predict(text)

    if return_label:
        label = evaluation > 0.5
        return evaluation, label
        
    return evaluation


def classify_with_gptzero(text: str, device: str, gptzero=None) -> float:
    if gptzero==None:
        classifier = "GPTZero"
        gptzero = prepare_classifier(classifier=classifier, device=device)
    
    evaluation = gptzero.get_score(text)

    return evaluation


def classify(query: str, response: str, classifiers: List[str], with_query: bool, device: str) -> str:
    assert response, "Response cannot be empty!"
    assert not with_query or query, "Query cannot be empty when 'Include Prompt' is selected!"
    assert len(classifiers) > 0, "At least one classifier must be selected!"

    text = query+response if with_query else response
    evaluations: List[float] = []
    classifiers_lower = [x.lower() for x in classifiers]

    if "binoculars" in classifiers_lower:
        evaluations.append(classify_with_binoculars(text, device))
    if "simpleai" in classifiers_lower:
        evaluations.append(classify_with_simpleai(text, device))
    if "ghostbuster" in classifiers_lower:
        evaluations.append(classify_with_ghostbuster(text, device))
    if "gptzero" in classifiers_lower:
        evaluations.append(classify_with_gptzero(text, device))

    return process_output(classifiers, evaluations)