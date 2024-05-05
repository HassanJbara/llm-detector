import os
from typing import List, Dict

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

openai_config = {
    "API_KEY": os.environ.get("OPENAI_API_KEY", None)
}

supported_detectors: List[Dict[str, str]] = [
    {
        "key": "Binoculars",
        "value": "binoculars"
    }, 
    {
        "key": "SimpleAI",
        "value": "Hello-SimpleAI/chatgpt-detector-roberta"
    },
    {
        "key": "Ghostbuster",
        "value": "ghostbuster"
    }
]