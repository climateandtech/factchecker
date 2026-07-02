from build.__main__ import _COLORS
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from pathlib import Path

#IMPORST FOR CSS
BASE_DIR = Path(__file__).resolve().parent / "styles"

css_files = [
    "styles.css",
    "colors.css",
    "fonts.css",
    "buttons.css",
    "cards/result_card.css",
    "cards/claim_card.css",
    "cards/factchecker_card.css",
]

css = ""

for file in css_files:
    with open(BASE_DIR / file) as f:
        css += f.read() + "\n"

# Just testfunction acts as dummy 
def claim(claim, scientific_source):
    information_integrity="Misleading"
    confidence="80%",
    reasoning="The claim partially ignores current knowlidge",
    evidence="table"
    
    return f"""
        <section class="result-card">
            <h2>Claim Result: <i class="pill-integrity">{information_integrity}</i> </h2> 

            <h4><i>87% confidence</i></h4>
            <br>
            <p>The claim is partially supported by current scientific evidence, but omits important contextual findings from IPCC reports.</p>
            <br>
            <br>
            <p>Backed with following sources:</p>
            <p>
                <strong class="pill-evidence">IPCC AR6</strong> 
                <strong class="pill-evidence">Climate Feedback</strong>
                <strong class="pill-evidence">Peer-reviewed</strong>
            </p>
        </section>
        """

with gr.Blocks(css=css) as demo:
    with gr.Row(elem_classes="bg-color"):
        with gr.Column(elem_classes="main-container"):
            # TOP
            with gr.Column():
                gr.HTML(
                    '''
                    <section>
                        <h1> Climate+Tech Factchecker</h1>
                        <p>Climate+Tech launches an open source AI fact-checking system that combines multiple AI models to evaluate climate claims against scientific sources, achieving 85% agreement with expert consensus.</p>
                    </section>
                    '''
                )

            # LEFT SIDE FOR USER CLAIM AND SOURCE
            with gr.Row(elem_classes="fact-checker-container"):
                with gr.Column(elem_classes="claim-card"):
                    claim_input = gr.TextArea(
                        label= "Claim",
                        placeholder="Enter your claim here...\n \n" \
                        "For example: 'The Earth atmosphere is perfectly fine and theses so called holes are only a product from the pharma inustry'",
                        container=False,
                        elem_classes="claim-text-area"
                    )
                    with gr.Accordion("Add Scientific souce", open=False):
                        source_input = gr.Textbox(
                            placeholder="https://example.source.com",
                            container=False
                        )

                    with gr.Group(elem_classes="btn-group"):
                        confirm_btn = gr.Button("Evaluate Claim", elem_classes="confirm-btn")
                        learn_btn = gr.Button("Learn More", elem_classes="learn-more-btn")
                    
                # RIGHT SIDE FOR RESULT
                with gr.Column():
                    result_output = gr.HTML(
                        '''
                        <section class="result-card">
                            <h2>Get your Claims evaluated</h2>
                            <br>
                            <section class="info-text">
                                <p><strong>Verify Climate Claims:</strong> Check statements about climate change against trustworthy sources like IPCC reports</p>
                                <br>
                                <p><strong>Verify Political Claims:</strong> Fact-check political statements and policy claims against authoritative sources and evidence</p>
                                <br>
                                <p><strong>Detect Greenwashing:</strong> Identify and verify green claims and potential misleading greenwashing</p>
                                <br>
                                <p><strong>Combat Disinformation:</strong> Provide journalistic fact checks for both climate and political disinformation</p>
                                <br>
                            </section>
                        </section>
                        '''
                    )
                    confirm_btn.click(fn=claim, inputs=[claim_input, source_input], outputs=result_output, api_name="confirm")
                    

demo.launch()