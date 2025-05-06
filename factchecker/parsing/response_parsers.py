"""Base class for response parsers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class ResponseParser(ABC):
    """Abstract base class for response parsers."""
    
    @abstractmethod
    def parse_verdict(self, response_content: str) -> Optional[str]:
        """Parse the verdict from the response content.
        
        Args:
            response_content (str): The raw response content from the LLM
            
        Returns:
            Optional[str]: The parsed verdict, or None if parsing failed
        """
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get any additional metadata from the last parsed response.
        
        Returns:
            Dict[str, Any]: Additional metadata extracted during parsing
        """
        return {} 