"""HyDE-style query generator for generating additional search queries."""

import logging
from typing import List, Optional
from dataclasses import dataclass
from factchecker.core.llm import load_llm
from llama_index.core.llms import ChatMessage
from factchecker.parsing.hyde_parser import HyDEQueryParser

logger = logging.getLogger(__name__)

@dataclass
class HyDEQueryConfig:
    """Configuration for HyDE query generation."""
    num_queries: int = 1
    max_query_length: int = 500  # Increased from 200 to 500 to allow for more detailed scientific queries
    temperature: float = 0.7
    system_prompt_template: str = """You are a {domain} expert evaluating scientific papers.
Your task is to write a hypothetical scientific passage that would help evaluate a claim.

Write a passage that:
1. Looks like it came from a real scientific paper
2. Contains specific numbers, measurements, or data
3. Uses proper technical terminology from your field
4. Would help verify or refute the claim

For example, if evaluating a claim about human CO2 emissions, write something like:
"Our measurements of human respiratory CO2 emissions indicate an average output of 1.2 kg per person per day. Analysis of global population data suggests that human respiration contributes approximately 3.4 gigatons of CO2 annually, representing 0.1% of total anthropogenic greenhouse gas emissions."

Format your response as:
<passage>Your hypothetical scientific passage here</passage>"""

    user_prompt_template: str = """Claim: {claim}

Write a hypothetical scientific passage that would help evaluate this claim.
The passage should:
- Look like it came from a real scientific paper
- Include specific measurements and data
- Use technical terminology from your domain
- Not exceed {max_length} characters

Format your response as:
<passage>Your hypothetical scientific passage here</passage>"""

class HyDEQueryGenerator:
    """Generates additional search queries using HyDE-style prompting when evidence is insufficient."""

    def __init__(
        self,
        config: Optional[HyDEQueryConfig] = None,
        llm = None
    ):
        """Initialize the HyDE query generator.
        
        Args:
            config: Configuration for query generation
            llm: Language model to use (optional)
        """
        self.config = config or HyDEQueryConfig()
        self.llm = llm if llm is not None else load_llm()
        self.parser = HyDEQueryParser()

    def generate_queries(
        self,
        claim: str,
        domain: str,
        domain_description: str
    ) -> List[str]:
        """Generate additional search queries for a claim.
        
        Args:
            claim: The claim to generate queries for
            domain: The domain of expertise (e.g., "Climate Science", "Biology")
            domain_description: Detailed description of the domain expertise
            
        Returns:
            List of generated search queries
        """
        queries = []
        logger.info(f"Generating HyDE queries for claim: {claim}")
        logger.info(f"Domain: {domain}")
        logger.info(f"Domain description: {domain_description}")
        logger.info(f"Configuration: num_queries={self.config.num_queries}, max_length={self.config.max_query_length}, temperature={self.config.temperature}")
        
        for i in range(self.config.num_queries):
            try:
                # Format system and user prompts
                system_prompt = self.config.system_prompt_template.format(
                    domain=domain_description
                )
                
                user_prompt = self.config.user_prompt_template.format(
                    claim=claim,
                    domain=domain,
                    max_length=self.config.max_query_length
                )
                
                logger.debug(f"Attempt {i+1}/{self.config.num_queries}")
                logger.debug(f"System prompt: {system_prompt}")
                logger.debug(f"User prompt: {user_prompt}")
                
                # Generate query
                messages = [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt)
                ]
                
                response = self.llm.chat(
                    messages,
                    temperature=self.config.temperature
                )
                
                logger.debug(f"Raw LLM response: {response.message.content}")
                
                # Parse query from XML response
                query = self.parser.parse_response(response.message.content.strip())
                
                if query:
                    # Validate query length
                    if len(query) > self.config.max_query_length:
                        logger.warning(f"Query exceeded max length ({len(query)} > {self.config.max_query_length})")
                        query = query[:self.config.max_query_length]
                        logger.info(f"Truncated hypothetical evidence: {query}")
                    
                    queries.append(query)
                    logger.info(f"Generated hypothetical evidence {i+1}:\n{query}")
                else:
                    logger.warning(f"Failed to parse evidence from response: {response.message.content}")
                
            except Exception as e:
                logger.error(f"Error generating HyDE query {i+1}: {str(e)}")
                continue
        
        logger.info(f"Generated {len(queries)} HyDE queries successfully")
        return queries 