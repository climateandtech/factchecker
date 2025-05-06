"""Parser for HyDE query responses."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

class HyDEQueryParser:
    """Parser for extracting passages from HyDE responses in XML format."""
    
    def __init__(self):
        self._passage_pattern = re.compile(r'<passage>(.*?)</passage>', re.DOTALL)
    
    def parse_response(self, response_content: str) -> Optional[str]:
        """Parse a passage from the response content.
        
        Args:
            response_content: The raw response content containing XML tags
            
        Returns:
            The extracted passage if found, None otherwise
        """
        try:
            # Extract passage
            passage_match = self._passage_pattern.search(response_content)
            if not passage_match:
                logger.warning("No passage found in XML response")
                return None
            
            passage = passage_match.group(1).strip()
            return passage
            
        except Exception as e:
            logger.error(f"Error parsing HyDE response: {str(e)}")
            return None 