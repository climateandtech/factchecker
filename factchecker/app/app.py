import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from llama_index.core import Settings

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback_prompts import (
    advocate_primer,
    arbitrator_primer,
)
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.tools.sources_downloader import download_pdf

# Only enable debugpy if DEBUG environment variable is set
if os.getenv("DEBUG"):
    import debugpy
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()

from factchecker.utils.logging_config import setup_logging

logger = setup_logging()



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

def build_indexer_retriever_configs_from_base_path_sources(
    base_path_sources: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """
    Build indexer-retriever configurations by detecting subfolders (or fallback to base folder).

    Args:
        base_path_sources (str): Path to the base folder containing subfolders or files.
        chunk_size (int): Chunk size for text splitting.
        chunk_overlap (int): Overlap between text chunks.
        top_k (int): Top-k similarity retrieval setting.
        min_score (float): Minimum similarity score for retrieval.

    Returns:
        List[dict]: List of indexer_retriever config dictionaries.
    """
    configs = []

    base_path_sources = Path(base_path_sources)

    if not base_path_sources.exists() or not base_path_sources.is_dir():
        raise ValueError(f"Base data path {base_path_sources} does not exist or is not a directory.")

    subfolders = [f for f in sorted(base_path_sources.iterdir()) if f.is_dir()]

    if subfolders:
        logger.info(f"Found {len(subfolders)} source folders under {base_path_sources}.")

        for folder in subfolders:
            logger.info(f"Adding source: {folder.name}")

            configs.append({
                "indexer_options": {
                    "source_directory": str(folder),
                    "index_name": f"{folder.name}_index",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
                "retriever_options": {
                    "similarity_top_k": top_k,
                    "min_score": min_score,
                },
            })

    else:
        logger.warning(f"No subfolders found. Using base directory {base_path_sources} as a single source.")

        configs.append({
            "indexer_options": {
                "source_directory": str(base_path_sources),
                "index_name": f"{base_path_sources.name}_index",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "retriever_options": {
                "similarity_top_k": top_k,
                "min_score": min_score,
            },
        })

    return configs

def get_strategy(
    base_path_sources: str = "data/sources",
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    top_k: int = 8,
    min_score: float = 0.75,
    max_evidences: int = 10,
    advocate_temperature: float = 0.0,
    mediator_temperature: float = 0.0
) -> AdvocateMediatorStrategy:
    """
    Create an instance of the AdvocateMediatorStrategy.

    This function builds a complete AdvocateMediatorStrategy by detecting available
    source folders or falling back to the base source path directly if no subfolders are found.
    
    Args:
        base_path_sources (str, optional): Base directory path containing source documents or subfolders. 
            Defaults to "data/sources".
        chunk_size (int, optional): Size of text chunks for indexing. Defaults to 150.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 20.
        top_k (int, optional): Number of top retrieval results to use. Defaults to 8.
        min_score (float, optional): Minimum similarity score for retrieval. Defaults to 0.75.
        max_evidences (int, optional): Maximum number of evidence pieces per advocate. Defaults to 10.
        advocate_temperature (float, optional): Temperature setting for the advocate LLM. Defaults to 0.0.
        mediator_temperature (float, optional): Temperature setting for the mediator LLM. Defaults to 0.0.

    Returns:
        AdvocateMediatorStrategy: An initialized strategy ready to evaluate claims.
    """
    logger.info("Creating AdvocateMediatorStrategy with following settings:")
    for param, value in locals().items():
        logger.info(f"  - {param}: {value}")

    indexer_retriever_configs = build_indexer_retriever_configs_from_base_path_sources(
        base_path_sources=base_path_sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_score=min_score,
    )

    evidence_options = {}

    advocate_options = {
        'max_evidences': max_evidences,
        'top_k': top_k,
        'min_score': min_score,
        'temperature': advocate_temperature
    }
    
    mediator_options = {
        'temperature': mediator_temperature
    }

    return AdvocateMediatorStrategy(
        indexer_retriever_configs=indexer_retriever_configs,
        evidence_options=evidence_options,
        advocate_options=advocate_options,
        mediator_options=mediator_options,
    )

def on_validate(claim: str, sources: list):
    logger.info(f"🔍 Processing claim: {claim}")
    logger.info(f"🔍 Selected sources: {sources}")
    try:
        if not claim or not claim.strip():
            raise ValueError("Please enter a claim to validate")
            
        if not sources:  # Add this check
            raise ValueError("Please select at least one source for fact-checking")
        
        strategy = get_strategy(advocate_sources=sources)
        final_verdict, mediator_reasoning, verdicts, reasonings, evidences = strategy.evaluate_claim(claim.strip())
        
        # Format advocate data for the table display
        advocate_data = []
        for i, (verdict, reasoning, evidence) in enumerate(zip(verdicts, reasonings, evidences)):
            advocate_data.append([
                f"LLM {i+1}",  # Advocate column
                verdict,        # Verdict column
                reasoning,      # Reasoning column
                "\n".join(evidence) if evidence else ""  # Evidence column
            ])
        
        return (
            final_verdict,
            mediator_reasoning,
            advocate_data      # Now passing formatted data for the table
        )
    except Exception as e:
        logger.error(f"Error in on_validate: {str(e)}")
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

def download_source(url: str) -> tuple[str, list[dict], list[str]]:
    """
    Downloads content and returns status, updated sources list, and updated choices
    """
    try:
        if not url:
            return "Please enter a URL", get_available_sources(), get_source_choices()
            
        filename = url.split('/')[-1] or "downloaded_content.txt"
        os.makedirs("data", exist_ok=True)
        
        download_pdf(url, "data", filename)
        
        return (
            f"Successfully downloaded {filename}", 
            get_available_sources(),  # Update sources table
            get_source_choices()      # Update advocate choices
        )
    except Exception as e:
        return f"Error downloading source: {str(e)}", get_available_sources(), get_source_choices()

def get_source_choices():
    """Get list of available source names from data directory"""
    try:
        data_dir = Path("data")
        if data_dir.exists():
            return [f.stem for f in data_dir.glob("*") if f.is_file()]
        return ["ipcc_ar6_wg1"]  # Updated default fallback
    except Exception as e:
        logger.error(f"Error getting source choices: {e}")
        return ["ipcc_ar6_wg1"]  # Updated default fallback

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

            with gr.Tab("Settings & Sources"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Retrieval Settings")
                        chunk_size = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=150,
                            step=50,
                            label="Chunk Size",
                            info="Size of text chunks for processing"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=20,
                            step=10,
                            label="Chunk Overlap",
                            info="Overlap between chunks"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=8,
                            step=1,
                            label="Top K Results"
                        )
                        min_score = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Minimum Similarity Score"
                        )

                    with gr.Column():
                        gr.Markdown("### Advocate Settings")
                        advocate_sources = gr.CheckboxGroup(
                            choices=get_source_choices(),  # Dynamic choices from data directory
                            label="Select Advocates",
                            info="Choose which sources to use as advocates",
                        )
                        max_evidences = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Max Evidence per Advocate",
                            info="Maximum number of evidence chunks per advocate"
                        )
                        advocate_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Advocate Temperature",
                            info="Temperature for advocate LLM responses"
                        )

                    with gr.Column():
                        gr.Markdown("### Mediator Settings")
                        mediator_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Mediator Temperature",
                            info="Temperature for mediator LLM responses"
                        )
                        
                # Add a status message component
                settings_status = gr.Text(label="Status")  # Remove visible=False to make it always visible
                apply_settings = gr.Button("Apply Settings")
                
                apply_settings.click(
                    fn=lambda cs, co, tk, ms, as_, me, at, mt: (
                        get_strategy(
                            chunk_size=cs,
                            chunk_overlap=co,
                            top_k=tk,
                            min_score=ms,
                            advocate_sources=as_,
                            max_evidences=me,
                            advocate_temperature=at,
                            mediator_temperature=mt
                        ),
                        "✅ Settings applied successfully!"  # Add emoji for better visibility
                    ),
                    inputs=[
                        chunk_size, chunk_overlap, top_k, min_score,
                        advocate_sources, max_evidences,
                        advocate_temperature, mediator_temperature
                    ],
                    outputs=[gr.Textbox(visible=False), settings_status]  # Keep status visible
                )

            with gr.Tab("Review & Validate"):
                with gr.Row():
                    with gr.Column():
                        review_claim = gr.Textbox(
                            label="Claim to Review",
                            placeholder="Enter or select a claim to review..."
                        )
                        
                        with gr.Accordion("AI Analysis", open=True):
                            ai_verdict = gr.Textbox(
                                label="AI Verdict",
                                interactive=False
                            )
                            ai_reasoning = gr.Textbox(
                                label="AI Reasoning",
                                lines=3,
                                interactive=False
                            )
                            
                        with gr.Accordion("Evidence Review", open=True):
                            evidence_table = gr.Dataframe(
                                headers=["Evidence", "Source", "Relevance", "Your Rating"],
                                interactive=True
                            )
                            
                        with gr.Row():
                            accept_btn = gr.Button("Accept AI Analysis")
                            modify_btn = gr.Button("Provide Custom Analysis")
                            
                    with gr.Column(visible=False) as custom_analysis:
                        custom_verdict = gr.Radio(
                            choices=["correct", "incorrect", "misleading", "unsupported"],
                            label="Your Verdict"
                        )
                        custom_reasoning = gr.Textbox(
                            label="Your Reasoning",
                            lines=3
                        )
                        submit_analysis_btn = gr.Button("Submit Analysis")

        # Event handlers
        validate_btn.click(
            fn=lambda claim, sources: on_validate(claim, sources),
            inputs=[claim_input, advocate_sources],
            outputs=[final_verdict, mediator_reasoning, advocate_outputs]
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

def get_available_sources():
    """
    Returns a list of available sources in the data directory
    """
    try:
        sources = []
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob("*"):
                if file.is_file():
                    sources.append([
                        file.name,
                        "Active",
                        datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    ])
        return sources if sources else [["No sources available", "-", "-"]]
    except Exception as e:
        logger.error(f"Error getting available sources: {e}")
        return [["Error loading sources", "Error", "-"]]

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(show_api=False)
