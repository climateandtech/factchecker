"""Specialized advocate implementation for climate check domains."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from factchecker.steps.advocate import AdvocateStep
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.core.llm import load_llm
from factchecker.experiments.advocate_mediator_climatecheck.reranker_utils import BGEReranker
from factchecker.clustering.paper_clustering import PaperClusterer

@dataclass
class ClimateExpertise:
    """Represents a climate-related domain of expertise."""
    name: str
    description: str
    keywords: List[str]
    prompt_template: str

# Pre-defined climate expertise domains
CLIMATE_DOMAINS = [
    ClimateExpertise(
        name="Climate Science",
        description="Expert in climate systems, atmospheric science, and global warming mechanisms",
        keywords=["climate", "temperature", "greenhouse", "emissions", "atmosphere"],
        prompt_template="""You are a climate science expert specializing in {domain_description}.
Your expertise covers: {domain_keywords}

Evaluate claims based on your deep understanding of climate systems and atmospheric processes.
Consider:
- Physical mechanisms of climate change
- Atmospheric composition and dynamics
- Temperature and weather patterns
- Greenhouse gas effects and measurements
"""
    ),
    ClimateExpertise(
        name="Environmental Impact",
        description="Expert in environmental consequences and ecosystem effects of climate change",
        keywords=["ecosystem", "biodiversity", "environmental", "impact", "species"],
        prompt_template="""You are an environmental impact expert specializing in {domain_description}.
Your expertise covers: {domain_keywords}

Evaluate claims based on your deep understanding of environmental systems and climate impacts.
Consider:
- Ecosystem responses to climate change
- Biodiversity impacts
- Environmental degradation
- Species adaptation and migration
"""
    ),
    ClimateExpertise(
        name="Climate Policy",
        description="Expert in climate policy, mitigation strategies, and international agreements",
        keywords=["policy", "agreement", "mitigation", "regulation", "governance"],
        prompt_template="""You are a climate policy expert specializing in {domain_description}.
Your expertise covers: {domain_keywords}

Evaluate claims based on your deep understanding of climate policy and governance.
Consider:
- International climate agreements
- Policy effectiveness
- Mitigation strategies
- Regulatory frameworks
"""
    )
]

class ClimateSpecializedAdvocate(AdvocateStep):
    """An advocate specialized in a particular climate-related domain."""
    
    def __init__(
        self,
        retriever: AbstractRetriever,
        domain_expertise: ClimateExpertise,
        llm=None,
        options: Optional[Dict[str, Any]] = None,
        evidence_options: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
        use_clustering: bool = True
    ):
        """Initialize a climate-specialized advocate.
        
        Args:
            retriever: Retriever instance for evidence gathering
            domain_expertise: Climate domain expertise configuration
            llm: Language model to use (optional)
            options: General advocate options
            evidence_options: Evidence gathering options
            use_reranker: Whether to use BGE reranker for paper selection
            use_clustering: Whether to use domain clustering for paper selection
        """
        # Extract HyDE config from options if present
        hyde_config = options.pop('hyde_config', None) if options else None
        
        # Initialize base advocate
        super().__init__(
            retriever=retriever,
            llm=llm,
            options=options,
            evidence_options=evidence_options,
            hyde_config=hyde_config
        )
        
        self.domain = domain_expertise
        self.use_reranker = use_reranker
        self.use_clustering = use_clustering
        
        if use_reranker:
            self.reranker = BGEReranker()
        
        if use_clustering:
            self.clusterer = PaperClusterer(use_bge=True)
        
        # Override system prompt with domain-specific template
        self.system_prompt = domain_expertise.prompt_template.format(
            domain_name=domain_expertise.name,
            domain_description=domain_expertise.description,
            domain_keywords=", ".join(domain_expertise.keywords)
        )
    
    def _get_relevant_papers(self, claim: str, top_k: int = 5) -> List[str]:
        """Get papers relevant to the claim, using reranking and/or clustering.
        
        Args:
            claim: The claim to find papers for
            top_k: Number of papers to return
            
        Returns:
            List of relevant paper texts
        """
        # Get initial papers from retriever
        papers = self.retriever.retrieve(claim)
        
        if self.use_clustering:
            # First cluster papers by domain
            domain_descriptions = [
                f"This paper discusses {domain.description.lower()}" 
                for domain in CLIMATE_DOMAINS
            ]
            domain_names = [domain.name for domain in CLIMATE_DOMAINS]
            
            clusters = self.clusterer.cluster_papers(
                papers=papers,
                domain_descriptions=domain_descriptions,
                domain_names=domain_names
            )
            
            # Get papers from our domain of expertise
            domain_papers = clusters.domain_papers[self.domain.name]
            
            # If we don't have enough papers in our domain, add highest scoring papers from other domains
            if len(domain_papers) < top_k:
                other_papers = [p for name, papers in clusters.domain_papers.items() 
                              for p in papers if name != self.domain.name]
                domain_papers.extend(other_papers[:top_k - len(domain_papers)])
            
            papers = domain_papers[:top_k]
        
        if self.use_reranker:
            # Further rerank the papers
            reranked_papers = self.reranker.rerank_papers(
                query=claim,
                papers=papers,
                top_k=top_k
            )
            return [paper for paper, _ in reranked_papers]
        
        return papers[:top_k]
    
    def evaluate_claim(self, claim: str) -> tuple[str, str]:
        """Evaluate a climate-related claim from the perspective of the advocate's domain expertise.
        
        Args:
            claim: The claim to evaluate
            
        Returns:
            A tuple of (verdict, reasoning) where the reasoning emphasizes domain expertise
        """
        # Get relevant papers using reranking and/or clustering
        papers = self._get_relevant_papers(claim)
        
        # Create domain info for HyDE
        domain_info = {
            'name': self.domain.name,
            'description': self.domain.description,
            'keywords': self.domain.keywords
        }
        
        # Get base evaluation using the selected papers and domain info
        verdict, base_reasoning = super().evaluate_claim(claim, domain_info)
        
        # Enhance reasoning with domain context
        domain_context = f"From the perspective of {self.domain.name} expertise: "
        enhanced_reasoning = domain_context + base_reasoning
        
        return verdict, enhanced_reasoning 