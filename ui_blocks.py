import gradio as gr
from typing import List, Callable, Dict

def action_buttons(func, inputs, outputs):
    with gr.Row() as row:
        submit_btn = gr.Button("Submit", variant="primary")
        reset_btn = gr.Button("Reset", variant="stop")

        submit_btn.click(fn=func, inputs=inputs, outputs=outputs)

    return row

def input_options(supported_detectors: List[Dict[str, str]]):
    with gr.Group():
        prompt_box = gr.Textbox(lines=2, label="Prompt")
        use_prompt_checkbox = gr.Checkbox(label="Include Prompt", value=False,)

    response_box = gr.Textbox(lines=10, label="Response")
    detectors_dropdown = gr.Dropdown(
        label="Detectors", 
        choices=[x["key"] for x in supported_detectors], 
        # multiselect=True,
        value=supported_detectors[0]["key"], 
        info="What detectors to use for evaluation"
    )

    return [prompt_box, response_box, detectors_dropdown, use_prompt_checkbox]

def input_area(func: Callable[[str, str, str], float],  supported_detectors: List[Dict[str, str]], outputs,):
    inputs = input_options(supported_detectors=supported_detectors)
    action_buttons(func, inputs, outputs)

def results_labels_html(labels: List[str], scores: List[float]):
    container_style = """
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        border-radius: 10px; 
        background-color: #1f2937; 
        padding: 10px;
    """
    item_style = """
        display: flex;
        justify-content: space-between;
        width: 100%;
        padding: 10px;
        margin: 10px;
        font-size: 16px;
        font-weight: bold;
        border: 0.5px solid gray;
        border-radius: 10px;
    """

    html_result = f"""
        <div style="{container_style}">
            <h2>Is the text AI generated? 🤖</h2>
            $PLACEHOLDER$
        </div>
    """

    label_score_html = ""

    for label, score in zip(labels, scores):
            label_score_html += f"""
                <div style='{item_style}'>
                    <span style='font-size: 16px; font-weight: bold;'>{label}</span>
                    <span style='font-size: 16px; font-weight: bold;'>{score}%</span>
                </div>
            """

    html_result = html_result.replace("$PLACEHOLDER$", label_score_html)

    return html_result