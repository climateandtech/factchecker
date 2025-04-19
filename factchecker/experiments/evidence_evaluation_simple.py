from factchecker.strategies.evidence_evaluation import EvidenceEvaluationStrategy

def main():
    # Define the options for each component
    indexer_options = {
        'source_directory': 'data',
        # Add other indexer options here
    }
    retriever_options = {
        'top_k': 4,
        # Add other retriever options here
    }
    evidence_options = {
        'min_score': 0.75,
        'pro_query_template': "evidence supporting: {claim}",
        'contra_query_template': "evidence refuting: {claim}",
        # Add other evidence step options here
    }
    evaluate_options = {
        'pro_prompt_template': "Pro evidence: {evidence}",
        'con_prompt_template': "Con evidence: {evidence}",
        'system_prompt_template': "You are assessing the following claim based on pro and con evidence: {claim}. Provide your reasoning step by step. In the end FINAL VERDICT, claim is supported by evidence because, claim is contradicted by evidence because or i dont have enough information."
        # Add other evaluate step options here
    }

    # Initialize the evidence evaluation strategy with the options
    strategy = EvidenceEvaluationStrategy(indexer_options, retriever_options, evidence_options, evaluate_options)

    # Define the claim to evaluate
    claim = "Climate change would have happened without humans"

    # Evaluate the claim
    result, pro_count, contra_count, pro_evidence, contra_evidence = strategy.evaluate_claim(claim)

    # Print the results
    print(f"Claim: {claim}")
    print(f"Result: {result}")
    print(f"Pro evidence count: {pro_count}")
    print(f"Contra evidence count: {contra_count}")

    # Additional printing of evidence details
    print("\nPro Evidence:")
    for evidence in pro_evidence:
        print(evidence)
    
    print("\nCon Evidence:")
    for evidence in contra_evidence:
        print(evidence)

if __name__ == "__main__":
    main()