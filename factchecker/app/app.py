from build.__main__ import _COLORS
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

theme = gr.themes.Default(primary_hue="blue").set(
    background_fill_primary='#2C2C2C',
    body_text_color= '#AAAAAA',
    loader_color="#4A3131",
    slider_color="#0097B5",
)

climate_green = '#00965A',
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

# class ClimateTheme(Base):
#     def __init__(
#         self,
#         *,
#         primary_hue: colors.Color | str = colors.green,
#         secondary_hue: colors.Color | str = colors.blue,
#         neutral_hue: colors.Color | str = colors.gray,
#         spacing_size: sizes.Size | str = sizes.spacing_md,
#         radius_size: sizes.Size | str = sizes.radius_md,
#         text_size: sizes.Size | str = sizes.text_md,
#         font: fonts.Font | str | list[fonts.Font | str] = (
#             "Helvetica Neue",
#             "ui-sans-serif",
#             "system-ui",
#             "sans-serif",
#         ),
#     ):
#         super().__init__(
#             primary_hue=primary_hue,
#             secondary_hue=secondary_hue,
#             neutral_hue=neutral_hue,
#             spacing_size=spacing_size,
#             radius_size=radius_size,
#             text_size=text_size,
#             font=font,
#         )
        
#         self.name = "climate_theme"
        
#         # Set both light and dark mode colors
#         self.set(
#             # Light mode colors (default)
#             body_background_fill="white",
#             block_background_fill="white",
#             block_label_background_fill="white",
#             input_background_fill="white",
#             button_primary_background_fill="*primary_500",
#             button_primary_text_color="white",
#             block_title_text_color="black",
#             block_label_text_color="black",
            
#             # Dark mode colors
#             body_background_fill_dark="*neutral_950",
#             block_background_fill_dark="*neutral_900",
#             block_label_background_fill_dark="*neutral_900",
#             input_background_fill_dark="*neutral_900",
#             button_primary_background_fill_dark="*primary_500",
#             button_primary_text_color_dark="white",
#             block_title_text_color_dark="white",
#             block_label_text_color_dark="white"
#         )
#         def getClimateThemes(self):
#             pass

def claim(claim, scientific_source):
    information_integrity="misleading"
    confidence="80%",
    reasoning="The claim partially ignores current knowlidge",
    evidence="table"
    
    return f"""
# Claim Result

**Integrity:** {information_integrity}

**Confidence:** 0.87

## Reasoning
This claim is partially supported by scientific evidence.
"""

# demo = gr.Interface(
#     fn=claim,
#     inputs=[gr.Textbox(label="Left", lines=3, info="abggc"), gr.State(value=[])],
#     outputs=["json", gr.State()],
#     title="Climate+Tech Factchecker",
# )

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
            '''
            # Climate+Tech Factchecker 
            
            Climate+Tech launches an open source AI fact-checking system that combines multiple AI models to evaluate climate claims against scientific sources, achieving 85% agreement with expert consensus.
            '''
            
        )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "Claim"
            )
            claim_input = gr.Textbox(
                label= "Claim",
                placeholder="Enter your claim...",
                container=False
            )
            with gr.Accordion("Add Scientific souce", open=False):
                source_input = gr.Textbox(
                    placeholder="https://example.source.com",
                    container=False
                )
            confirm_btn = gr.Button("Confirm")
            
        
        with gr.Column():
            result_output = gr.Markdown(
                '''
                # Your result of your checked claim
                
                '''
            )
            # with gr.Group():
            #     information_integrity=gr.Textbox(label=None)	
            #     confidence=gr.Number()
            #     reasoning=gr.Textbox()
            #     evidence=()
            
            confirm_btn.click(fn=claim, inputs=[claim_input, source_input], outputs=result_output, api_name="confirm")
            

demo.launch(theme= theme)