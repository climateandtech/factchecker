"""Example demonstrating specialized climate advocates."""

from factchecker.experiments.advocate_mediator_climatecheck.specialized_advocate import (
    ClimateSpecializedAdvocate,
    CLIMATE_DOMAINS
)
from factchecker.experiments.advocate_mediator_climatecheck.climatecheck_sources import get_test_documents
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.steps.mediator import MediatorStep

def main():
    # Get test documents
    documents = get_test_documents()
    
    # Create indexer and retriever
    indexer = LlamaVectorStoreIndexer({"index_name": "test_climate", "documents": documents})
    retriever = LlamaBaseRetriever(indexer=indexer, options={"top_k": 10})  # Increased top_k for reranking
    
    # Create specialized advocates for each domain
    advocates = []
    for domain in CLIMATE_DOMAINS:
        advocate = ClimateSpecializedAdvocate(
            retriever=retriever,
            domain_expertise=domain,
            options={
                "label_options": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            },
            evidence_options={"min_score": 0.7},
            use_reranker=True  # Enable reranking
        )
        advocates.append(advocate)
    
    # Create mediator
    mediator = MediatorStep(options={"system_prompt": "You are a climate fact-checking mediator..."})
    
    # Example claim to evaluate
    claim = """
    Rising global temperatures have led to increased frequency and intensity of extreme weather events,
    particularly affecting vulnerable ecosystems and communities, while also straining international
    climate policy frameworks.
    """
    
    # Get evaluations from each specialized advocate
    results = []
    for advocate in advocates:
        verdict, reasoning = advocate.evaluate_claim(claim)
        results.append((verdict, reasoning))
        print(f"\nAdvocate ({advocate.domain.name}) evaluation:")
        print(f"Verdict: {verdict}")
        print(f"Reasoning: {reasoning}")
    
    # Combine through mediator
    final_verdict = mediator.synthesize_verdicts(results, claim)
    print("\nFinal Verdict:")
    print(final_verdict)

if __name__ == "__main__":
    main() 