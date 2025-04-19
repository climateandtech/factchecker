
from pydantic import BaseModel, Field, model_validator


class LabelOption(BaseModel):
    """A fact checking label choice including the definition."""

    label: str = Field(..., description="The label for the claim")
    definition: str = Field("", description="The definition of the label")