
from pydantic import BaseModel, Field, model_validator


class LabelOption(BaseModel):
    """A fact checking label choice including the definition."""

    label: str = Field(..., description="The label for the claim")
    definition: str = Field("", description="The definition of the label")

class ClaimFactCheck(BaseModel):
    """A fact-check result for a claim."""
    
    claim: str = Field(..., description="The claim being fact-checked")
    label: str = Field(..., description="The label for the claim")
    label_options: list[LabelOption] = Field(..., min_length=2, description="The available label options")
    reasoning: str = Field(..., description="The reasoning behind the verdict")