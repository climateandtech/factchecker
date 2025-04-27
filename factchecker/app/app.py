import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from llama_index.core import Settings

from factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback_prompts import (
    advocate_primer,
    arbitrator_primer,
)
from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.tools.sources_downloader import SourcesDownloader

# Only enable debugpy if DEBUG environment variable is set
if os.getenv("DEBUG"):
    import debugpy
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()

from factchecker.utils.logging_config import setup_logging

logger = setup_logging()

strategy: Optional[AdvocateMediatorStrategy] = None
sources_downloader = SourcesDownloader(output_folder="data/sources")

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

def apply_settings_and_create_strategy(
    advocate_subfolders: list[str],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_score: float,
    max_evidences: int,
    advocate_temperature: float,
    mediator_temperature: float,
) -> str:
    """
    Apply the current settings to create a new AdvocateMediatorStrategy.

    Args:
        advocate_subfolders (list[str]): The subfolders for the individual advocate sources.
        chunk_size (int): Text chunk size.
        chunk_overlap (int): Overlap between chunks.
        top_k (int): Number of top retrieval results.
        min_score (float): Minimum similarity score for retrieval.
        max_evidences (int): Maximum evidence pieces per advocate.
        advocate_temperature (float): Temperature for advocate LLM.
        mediator_temperature (float): Temperature for mediator LLM.

    Returns:
        str: Status message indicating success.
    """
    global strategy

    logger.info("Applying settings to create new strategy with parameters:")
    for param, value in locals().items():
        logger.info(f"  - {param}: {value}")

    strategy = get_strategy(
        advocate_subfolders=advocate_subfolders,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_score=min_score,
        max_evidences=max_evidences,
        advocate_temperature=advocate_temperature,
        mediator_temperature=mediator_temperature
    )

    global STRATEGY
    STRATEGY = strategy

    return "✅ Settings applied successfully!"


def build_indexer_retriever_configs_from_base_path_sources(
    base_path_sources: str,
    advocate_subfolders: Optional[list[str]],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """
    Build indexer-retriever configuration dictionaries based on a base source directory.

    This function detects source folders under a given base directory and creates 
    indexer-retriever configurations for them. If specific subfolders are provided, 
    only those will be included; otherwise, all subfolders will be processed. 
    If no valid subfolders are found, the base directory itself is used.

    Args:
        base_path_sources (str): Path to the base directory containing source folders.
        advocate_subfolders (Optional[List[str]]): List of specific subfolder names to include. 
            If None, all available subfolders will be used.
        chunk_size (int): Size of text chunks for indexing.
        chunk_overlap (int): Overlap between text chunks for indexing.
        top_k (int): Number of top retrieval results.
        min_score (float): Minimum similarity score for retrieval.

    Returns:
        List[dict]: A list of indexer and retriever configuration dictionaries for the selected sources.
    """
    configs = []
    base_path = Path(base_path_sources)

    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"Base data path {base_path} does not exist or is not a directory.")

    if advocate_subfolders:
        logger.info(f"Using only selected subfolders: {advocate_subfolders}")
        folders_to_use = [base_path / folder_name for folder_name in advocate_subfolders]
    else:
        folders_to_use = [f for f in sorted(base_path.iterdir()) if f.is_dir()]
        logger.info(f"No subfolders selected explicitly. Using all {len(folders_to_use)} available subfolders.")

    if not folders_to_use:
        logger.warning(f"No valid folders found. Using base directory {base_path} as a single source.")
        folders_to_use = [base_path]

    for folder in folders_to_use:
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Folder {folder} does not exist or is not a directory. Skipping.")
            continue

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

    return configs

def get_strategy(
    base_path_sources: str = "data/sources",
    advocate_subfolders: Optional[list[str]] = None,
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
        advocate_subfolders: Specific selection of subfolders to use as Advocates.
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
        advocate_subfolders=advocate_subfolders,
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

def on_validate(claim: str):
    global strategy
    logger.info(f"Processing claim: {claim}")

    try:
        if not claim or not claim.strip():
            raise ValueError("Please enter a claim to validate")
            
        if strategy is None:
            raise ValueError("Please apply settings first!")
        
        final_verdict, mediator_reasoning, verdicts, reasonings, evidences = strategy.evaluate_claim(claim.strip())
        
        advocate_data = [
            [f"LLM {i+1}", verdict, reasoning, "\n".join(evidence) if evidence else ""]
            for i, (verdict, reasoning, evidence) in enumerate(zip(verdicts, reasonings, evidences, strict=True))
        ]

        return final_verdict, mediator_reasoning, advocate_data
    
    except Exception as e:
        logger.error(f"Error in on_validate: {str(e)}")
        return f"Error: {str(e)}", "", []


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
    Downloads a single source PDF from a URL and updates the UI source list.

    :param url: The url of the PDF to download
    """
    try:
        if not url:
            return "Please enter a URL", get_available_sources(), get_subfolder_choices()

        filename = url.split('/')[-1] or "downloaded_content.pdf"

        success = sources_downloader.download_pdf(
            url=url,
            output_folder="data/sources",
            output_filename=filename
        )
        
        if success:
            return (
                f"Successfully downloaded {filename}",
                get_available_sources(),
                get_subfolder_choices()
            )
        else:
            return (
                f"Failed to download {filename}",
                get_available_sources(),
                get_subfolder_choices()
            )
        
    except Exception as e:
        logger.error(f"Error in download_source: {str(e)}")
        return f"Error downloading source: {str(e)}", get_available_sources(), get_subfolder_choices()


def get_subfolder_choices(base_dir: str = "data/sources") -> list[str]:
    """
    Get a list of subfolder names inside the given base directory.

    Each subfolder typically represents a separate source collection for fact-checking.

    Args:
        base_dir (str, optional): The base directory to search for subfolders. Defaults to "data/sources".

    Returns:
        List[str]: A list of subfolder names relative to the base directory. 
                   Returns an empty list if no subfolders are found or an error occurs.
    """
    try:
        sources_dir = Path(base_dir)
        if sources_dir.exists():
            return [folder.name for folder in sources_dir.iterdir() if folder.is_dir()]
        return []
    except Exception as e:
        logger.error(f"Error getting subfolder choices from {base_dir}: {e}")
        return []


def get_individual_file_choices(base_dir: str = "data/sources") -> list[str]:
    """
    Get a list of individual PDF file paths inside the given base directory, searched recursively.

    This is useful for displaying all available source documents individually.

    Args:
        base_dir (str, optional): The base directory to search for PDF files. Defaults to "data/sources".

    Returns:
        List[str]: A list of file paths relative to the base directory for all PDFs found recursively.
                   Returns an empty list if no files are found or an error occurs.
    """
    try:
        sources_dir = Path(base_dir)
        if sources_dir.exists():
            return [
                str(file.relative_to(sources_dir))
                for file in sources_dir.rglob("*.pdf")
            ]
        return []
    except Exception as e:
        logger.error(f"Error getting individual file choices from {base_dir}: {e}")
        return []


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
                        advocate_subfolders = gr.CheckboxGroup(
                            choices=get_subfolder_choices(),  # Dynamic choices from data directory
                            label="Select Advocate Subfolders",
                            info="Choose which subfolders of your sources to use as advocates",
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
                    fn=apply_settings_and_create_strategy,
                    inputs=[
                        advocate_subfolders, chunk_size, chunk_overlap, top_k, min_score,
                        max_evidences, advocate_temperature, mediator_temperature
                    ],
                    outputs=[settings_status]
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
            fn=on_validate,
            inputs=[claim_input],
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
