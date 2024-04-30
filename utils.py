from typing import List

def greet(prompt: str, response: str, selection: List[str]) -> str:
    assert len(selection) > 0, "Please select at least one detector"
    
    return "Hello, " + prompt + selection[0]
