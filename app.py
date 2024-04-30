import gradio as gr
from utils import greet
from ui_blocks import input_area

supported_detectors = ["DetectGPT", "SimpleAI"]

# if __name__ == "__main__":
outputs_box = gr.Textbox(lines=10, label="Output")

with gr.Blocks() as demo:
    gr.Markdown("# LLM Detector Tools 🕵️‍♂️")
    with gr.Row():
        with gr.Column():
            input_area(greet, supported_detectors, outputs_box,)
        with gr.Column():
            outputs_box.render()
    
demo.launch()
    