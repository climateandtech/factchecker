import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.prompts.advocate_mediator_prompts import advocate_primer, arbitrator_primer
import pandas as pd
from datetime import datetime
import os
import json
from llama_index.core import Settings
from functools import lru_cache
import logging

# Brand colors from the image
COLORS = {
    'climate_green': '#00965A',
    'tech_blue': '#0046BE',
    'background': '#1F1F1F',
    'secondary_bg': '#2C2C2C',
    'text': '#FFFFFF',
    'text_secondary': '#AAAAAA',
    'orange': '#FF5733',
    'input_border': '#3F3F3F'
}

class ClimateTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.green,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | list[fonts.Font | str] = (
            "Helvetica Neue",
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
        )
        
        self.name = "climate_theme"
        
        # Set both light and dark mode colors
        self.set(
            # Light mode colors (default)
            body_background_fill="white",
            block_background_fill="white",
            block_label_background_fill="white",
            input_background_fill="white",
            button_primary_background_fill="*primary_500",
            button_primary_text_color="white",
            block_title_text_color="black",
            block_label_text_color="black",
            
            # Dark mode colors
            body_background_fill_dark="*neutral_950",
            block_background_fill_dark="*neutral_900",
            block_label_background_fill_dark="*neutral_900",
            input_background_fill_dark="*neutral_900",
            button_primary_background_fill_dark="*primary_500",
            button_primary_text_color_dark="white",
            block_title_text_color_dark="white",
            block_label_text_color_dark="white"
        )

#@lru_cache()
def get_strategy():
    """
    Creates and caches an instance of the AdvocateMediatorStrategy.
    Using lru_cache to ensure we only create one instance.
    """
    indexer_options_list = [
        {
            'source_directory': 'data',
            'index_name': 'advocate1_index'
        }
    ]

    retriever_options_list = [
        {
            'similarity_top_k': 8,
            'indexer_options': indexer_options_list[0]
        }
    ]

    advocate_options = {
        'max_evidences': 10,
        'top_k': 8,
        'min_score': 0.75
    }
    
    mediator_options = {}

    return AdvocateMediatorStrategy(
        indexer_options_list=indexer_options_list,
        retriever_options_list=retriever_options_list,
        advocate_options=advocate_options,
        mediator_options=mediator_options,
        advocate_prompt=advocate_primer,
        mediator_prompt=arbitrator_primer
    )

def on_validate(claim: str):
    try:
        if not claim or not claim.strip():
            raise ValueError("Please enter a claim to validate")
        
        strategy = get_strategy()
        final_verdict, verdicts, reasonings, evidences = strategy.evaluate_claim(claim.strip())
        
        logging.info(f"Verdict received from mediator: {final_verdict}")
        logging.info(f"Verdicts received: {verdicts}")
        logging.info(f"Reasonings received: {reasonings}")
        logging.info(f"Evidence chunks: {evidences}")
        # Format advocate results for display
        if verdicts and reasonings and len(reasonings) > 1:
            advocate_data = [
                [f"Advocate {i+1}", v, r, "\n".join(e)]  # Include evidence chunks
                for i, (v, r, e) in enumerate(zip(verdicts, reasonings[:-1], evidences))
            ]
        else:
            advocate_data = []
            logging.warning("No advocate verdicts or reasonings available")
        
        logging.info(f"Formatted advocate data: {advocate_data}")
        logging.info(f"Final verdict being returned to UI: {final_verdict}")
        
        return (
            final_verdict,
            reasonings[-1],  # Mediator reasoning
            advocate_data
        )
    except Exception as e:
        logging.error(f"Error in on_validate: {str(e)}")
        return (
            f"Error: {str(e)}",
            "",
            []  # Empty list for no results
        )

def save_validation(claim: str, verdict: str, reasoning: str, chunks: list):
    try:
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty")
        if not reasoning or not reasoning.strip():
            raise ValueError("Please provide your reasoning")
        if not chunks or len(chunks) == 0:
            raise ValueError("Please add at least one evidence chunk")
        
        validation_data = {
            "claim": claim.strip(),
            "verdict": verdict,
            "reasoning": reasoning.strip(),
            "chunks": chunks,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        os.makedirs("experiments/results/custom_validations", exist_ok=True)
        filename = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"experiments/results/custom_validations/{filename}", 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        return "Validation saved successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

def create_interface():
    theme = ClimateTheme()

    with gr.Blocks(theme=theme, css=CUSTOM_CSS) as demo:
        gr.Markdown("""
            # <span style="color: #00965A">CLIMATE</span><span style="color: #00965A">+</span><span style="color: #0046BE">TECH</span>
            # Fact Checker
        """)
        
        with gr.Tabs():
            with gr.Tab("Validate Claim"):
                with gr.Row():
                    with gr.Column(scale=2):
                        claim_input = gr.Textbox(
                            label="Enter your claim",
                            placeholder="Enter a climate-related claim to validate...",
                            lines=3
                        )
                        validate_btn = gr.Button("Validate Claim")

                    with gr.Column(scale=3):
                        final_verdict = gr.Textbox(
                            label="Final Verdict",
                            interactive=False
                        )
                        mediator_reasoning = gr.Textbox(
                            label="Mediator Reasoning",
                            lines=5,
                            interactive=False
                        )
                        
                        with gr.Accordion("Evidence & Reasoning", open=True):
                            advocate_outputs = gr.Dataframe(
                                headers=["Advocate", "Verdict", "Reasoning", "Evidence"],
                                label="Advocate Verdicts and Reasoning",
                                interactive=False,
                                wrap=True,
                                elem_classes="table-container",
                                column_widths=["80px", "80px", "250px", "250px"]
                            )

            with gr.Tab("Custom Validation"):
                with gr.Row():
                    with gr.Column():
                        claim_text = gr.Textbox(
                            label="Claim",
                            lines=2,
                            placeholder="Enter the claim to validate..."
                        )
                        custom_verdict = gr.Radio(
                            choices=["correct", "incorrect", "misleading", "unsupported", "needs_more_context"],
                            label="Your Verdict",
                            value="incorrect"
                        )
                        custom_reasoning = gr.Textbox(
                            label="Your Reasoning",
                            lines=4,
                            placeholder="Explain why you chose this verdict..."
                        )

                with gr.Row():
                    chunk_text = gr.Textbox(
                        label="Evidence Chunk",
                        lines=3,
                        placeholder="Enter a relevant text chunk..."
                    )
                    chunk_source = gr.Textbox(
                        label="Source",
                        placeholder="Where is this chunk from?"
                    )
                    chunk_relevance = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.75,
                        step=0.05,
                        label="Relevance Score"
                    )
                
                chunks_table = gr.Dataframe(
                    headers=["Text", "Source", "Relevance"],
                    label="Evidence Chunks"
                )
                
                save_validation_btn = gr.Button("Save Validation")
                save_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        validate_btn.click(
            fn=on_validate,
            inputs=[claim_input],
            outputs=[final_verdict, mediator_reasoning, advocate_outputs]
        )

        save_validation_btn.click(
            fn=save_validation,
            inputs=[claim_text, custom_verdict, custom_reasoning, chunks_table],
            outputs=[save_status]
        )

    return demo

CUSTOM_CSS = """
footer {display: none !important;}
.gradio-container {min-height: 0px !important;}

/* Improved table styling */
.table-container {
    width: 100% !important;
}

.table-container table td {
    max-width: 250px !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    padding: 8px !important;
    vertical-align: top !important;
}

/* Column-specific widths */
.table-container td:nth-child(1) {  /* Advocate column */
    width: 80px !important;
}

.table-container td:nth-child(2) {  /* Verdict column */
    width: 80px !important;
}
"""

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(show_api=False)
