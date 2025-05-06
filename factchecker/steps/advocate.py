import logging
import re

from llama_index.core.llms import ChatMessage
from typing import Optional, List, Dict, Any, Tuple

from factchecker.config.config import DEFAULT_LABEL_OPTIONS
from factchecker.core.llm import load_llm
from factchecker.datastructures import LabelOption
from factchecker.prompts.advocate_prompts import get_default_system_prompt, get_default_user_prompt
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.steps.evidence import EvidenceStep
from factchecker.parsing.response_parsers import ResponseParser
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_parser import ClimateCheckParser
from factchecker.steps.hyde_query_generator import HyDEQueryGenerator, HyDEQueryConfig

logger = logging.getLogger(__name__)

class AdvocateStep:
    """
    A step in the fact-checking process that acts as an advocate by evaluating claims based on evidence.
    
    This class retrieves relevant evidence for a claim and uses an LLM to evaluate whether the claim
    is supported by the evidence, producing a verdict and reasoning.

    Args:
        retriever (AbstractRetriever): Retriever instance to use for evidence retrieval.
        llm (TODO): Language model instance to use for evaluation. If None, loads default model.
        options (dict, optional): Configuration options for the advocate step.
        evidence_options (dict, optional): Configuration for evidence gathering including
        parser (ResponseParser, optional): Parser instance to use for parsing LLM responses.
            If None, uses the ClimateCheckParser.
    """

    def __init__(
            self, 
            retriever: AbstractRetriever,
            llm = None, # TODO: Add type hint
            options: dict = None,
            evidence_options: dict = None,
            parser: Optional[ResponseParser] = None,
            hyde_config: Optional[HyDEQueryConfig] = None
        ) -> None:
        """Initialize an AdvocateStep instance."""
        self.retriever = retriever
        self.llm = llm if llm is not None else load_llm()
        self.options = options if options is not None else {}
        self.evidence_options = evidence_options if evidence_options is not None else {}
        self.system_prompt = self.options.pop('system_prompt', get_default_system_prompt())
        self.label_options = self.options.pop('label_options', DEFAULT_LABEL_OPTIONS)
        self.max_retries = self.options.pop('max_retries', 3)
        self.chat_completion_options = self.options.pop('chat_completion_options', {})
        
        # Extract domain info from options if present
        self.domain_info = {
            'name': self.options.pop('name', 'General'),
            'description': self.options.pop('description', 'General fact-checking expert'),
            'keywords': self.options.pop('keywords', [])
        }
        
        # Initialize parser with min_paper_score if provided in options
        min_paper_score = self.options.pop('min_paper_score', 8.0)
        self.parser = parser if parser is not None else ClimateCheckParser(min_paper_score=min_paper_score)
        
        # Initialize EvidenceStep
        self.evidence_step = EvidenceStep(
            retriever=retriever,
            options={
                'min_score': 0.7,  # Keep high threshold for quality matches
                **self.evidence_options,
            }
        )
        
        # Initialize HyDE query generator
        self.hyde_generator = HyDEQueryGenerator(config=hyde_config, llm=llm)

    def retrieve_evidence(self, claim: str) -> list[str]:
        """
        Retrieve relevant evidence for a given claim.

        Args:
            claim (str): The claim for which to retrieve evidence.

        Returns:
            list[str]: A list of evidence pieces relevant to the claim.
        """
        # First try normal evidence retrieval with high threshold
        evidence_list = self.evidence_step.gather_evidence(claim)
        
        # If no good evidence found and we have domain info, try HyDE
        if not evidence_list and self.hyde_generator:
            logger.info(f"No evidence found above threshold, attempting HyDE query generation")
            
            # Generate additional queries
            hyde_queries = self.hyde_generator.generate_queries(
                claim=claim,
                domain=self.domain_info['name'],
                domain_description=self.domain_info['description']
            )
            
            # Try to get evidence using each HyDE query
            for query in hyde_queries:
                logger.info(f"Retrieving evidence using HyDE query: {query}")
                hyde_evidence = self.evidence_step.gather_evidence(query)
                if hyde_evidence:
                    evidence_list.extend(hyde_evidence)
                    logger.info(f"Found {len(hyde_evidence)} additional pieces of evidence using HyDE query")
        
        return evidence_list

    def evaluate_claim(self, claim: str) -> tuple[str, str]:
        """
        Evaluate a claim based on gathered evidence using the language model.

        Args:
            claim (str): The claim to evaluate.

        Returns:
            A tuple including the label and reasoning.
        """
        try:
            # Retrieve evidence for the claim
            evidence_list = self.retrieve_evidence(claim)
            
            logger.info(f"Retrieved {len(evidence_list)} pieces of evidence for claim: {claim}")
            if evidence_list:
                logger.info("Sample of evidence being passed to LLM:")
                for i, evidence in enumerate(evidence_list[:3]):  # Log first 3 pieces
                    logger.info(f"Evidence {i+1} preview: {evidence[:200]}...")

            # If no evidence is found, return NOT_ENOUGH_INFO
            if not evidence_list:
                logger.warning("No evidence found for claim")
                return "NOT_ENOUGH_INFO", "NO EVIDENCE FOUND"

            # Get paper scores from the LLM
            scoring_prompt = get_default_user_prompt(claim=claim, evidence=evidence_list, label_options=self.label_options)
            scoring_messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=scoring_prompt)
            ]

            # Get response and parse verdict
            response = self.llm.chat(scoring_messages, **self.chat_completion_options)
            response_content = response.message.content.strip()
            
            # Use the configured parser to extract the verdict
            verdict = self.parser.parse_verdict(response_content)
            if verdict:
                # Get the reasoning by removing all XML tags
                reasoning = re.sub(r'<[^>]+>.*?</[^>]+>', '', response_content).strip()
                logger.info(f"Successfully parsed verdict: {verdict}")
                return verdict, reasoning
            else:
                logger.warning(f"Parser failed to extract verdict. Response: {response_content}")
                return "NOT_ENOUGH_INFO", "Failed to parse verdict"

        except Exception as e:
            logger.error(f"Error evaluating claim: {str(e)}")
            return "NOT_ENOUGH_INFO", f"Error during evaluation: {str(e)}"