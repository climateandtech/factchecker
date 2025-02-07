

from enum import Enum
from pydantic import BaseModel, Field, field_validator

# Default fact-checking label options and descriptions
DEFAULT_LABEL_OPTIONS = {
    "TRUE": "The claim is factually correct based on the available evidence.",
    "FALSE": "The claim is incorrect based on the available evidence.",
    "NOT_ENOUGH_INFORMATION": "There is insufficient evidence to determine the claim's accuracy.",
}

