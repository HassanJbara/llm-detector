import gradio as gr
from utils import classify, supported_detectors
from ui_blocks import input_area

# def main():

outputs_labels = gr.HTML(label="Output", value="""
                         <div style="display: flex; 
                                    flex-direction: column; 
                                    align-items: center; 
                                    border-radius: 10px; 
                                    background-color: #1f2937; 
                                    padding: 10px;">
                            <h2>Is the text AI generated? 🤖</h2>
                         </div>
                         """
                         )

with gr.Blocks() as demo:
    gr.Markdown("# LLM Detector Tools 🕵️‍♂️")
    with gr.Row():
        with gr.Column():
            input_area(classify, supported_detectors, outputs_labels,)
        with gr.Column():
            # gr.Markdown("## Is the text AI generated? 🤖")
            outputs_labels.render()

demo.launch()

# if __name__ == "__main__":
#     main()