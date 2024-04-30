import gradio as gr
from typing import List

def action_buttons(func, inputs, outputs):
    with gr.Row() as row:
        submit_btn = gr.Button("Submit", variant="primary")
        reset_btn = gr.Button("Reset", variant="stop")

        submit_btn.click(fn=func, inputs=inputs, outputs=outputs)

    return row

def input_options(supported_detectors: List[str]):
    with gr.Group():
        prompt_box = gr.Textbox(lines=2, label="Prompt")
        use_prompt_checkbox = gr.Checkbox(label="Include Prompt", value=False,)

    response_box = gr.Textbox(lines=10, label="Response")
    detectors_dropdown = gr.Dropdown(
        label="Detectors", 
        choices=supported_detectors, 
        multiselect=True,
        value=supported_detectors[0], 
        info="What detectors to use for evaluation"
    )

    return [prompt_box, response_box, detectors_dropdown]

def input_area(func,  supported_detectors: List[str], outputs,):
    inputs = input_options(supported_detectors=supported_detectors)
    action_buttons(func, inputs, outputs)
