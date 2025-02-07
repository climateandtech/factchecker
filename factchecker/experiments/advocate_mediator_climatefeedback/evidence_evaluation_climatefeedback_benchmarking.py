from factchecker.strategies.evidence_evaluation import EvidenceEvaluationStrategy

import pandas as pd
from sklearn.metrics import classification_report

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
        'system_prompt_template': "You are assessing the following claim based on pro and con evidence: {claim}.",
        'format_prompt': "Respond with one of the labels: INCORRECT, CORRECT Strictly stick to the following JSON output format: {'label': 'LABEL', 'reasoning': 'incorrect because of this and that'}"
        # Add other evaluate step options here
    }

    # Initialize the evidence evaluation strategy with the options
    strategy = EvidenceEvaluationStrategy(indexer_options, retriever_options, evidence_options, evaluate_options)


    # Load the claims from the CSV file
    claims_df = pd.read_csv('datasets/Combined_Overview_Climate_Feedback_Claims.csv')
    
    # Make the number of claims to test configurable
    num_claims_to_test = 200  # Example value, can be changed as needed

    # Prepare lists to store true labels and predicted results
    true_labels = []
    predicted_results = []

    # Iterate over the claims
    for index, row in claims_df.iterrows():
        if index >= num_claims_to_test:
            break

        # Extract the claim and the true label
        claim = row['Claim']
        true_label = row['Climate Feedback']

        # Evaluate the claim using the strategy
        result, pro_count, contra_count, pro_evidence, contra_evidence = strategy.evaluate_claim(claim)

        # Append the results for later evaluation
        true_labels.append(true_label)
        predicted_results.append(result)

        # Print the results for each claim
        print(f"Claim: {claim}")
        print(f"Result: {result}")
        print(f"Pro evidence count: {pro_count}")
        print(f"Contra evidence count: {contra_count}")
        print("\nPro Evidence:")
        for evidence in pro_evidence:
            print(evidence)
        print("\nCon Evidence:")
        for evidence in contra_evidence:
            print(evidence)

    def map_verdict(verdict, level=2):
        """
        Map the verdict to a specified level of consolidation.
        
        Args:
        verdict (str): The original verdict.
        level (int): The level of consolidation (2, 5, or 7).
        
        Returns:
        str: The consolidated verdict.
        """
        if level == 7:
            mapping = {
                "incorrect": "incorrect",
                "inaccurate": "inaccurate",
                "imprecise": "imprecise",
                "misleading": "misleading",
                "flawed reasoning": "flawed_reasoning",
                "lacks context": "lacks_context",
                "unsupported": "unsupported",
                "correct but": "correct_but",
                "mostly correct": "mostly_correct",
                "mostly accurate": "mostly_accurate",
                "accurate": "accurate",
                "correct": "correct"
            }
        elif level == 5:
            mapping = {
                "incorrect": "incorrect",
                "inaccurate": "incorrect",
                "imprecise": "imprecise",
                "misleading": "misleading",
                "flawed reasoning": "flawed_reasoning",
                "lacks context": "unsupported",
                "unsupported": "unsupported",
                "correct but": "mostly_correct",
                "mostly correct": "mostly_correct",
                "mostly accurate": "correct",
                "accurate": "correct",
                "correct": "correct"
            }
        elif level == 2:
            correct_labels = [
                "correct", "mostly_correct", "correct_but", "mostly_accurate", "accurate"
            ]
            incorrect_labels = [
                "incorrect", "inaccurate", "unsupported", "misleading", "flawed_reasoning", "lacks_context", "mostly_inaccurate", "imprecise"
            ]
            if verdict in correct_labels:
                return "correct"
            elif verdict in incorrect_labels:
                return "incorrect"
            else:
                return "unknown"
        else:
            return "unknown"

        return mapping.get(verdict, "unknown")

    # Example usage:
    example_verdicts = ["correct", "inaccurate", "unsupported", "mostly accurate", "imprecise"]
    mapped_verdicts_seven = [map_verdict(verdict, level=7) for verdict in example_verdicts]
    mapped_verdicts_five = [map_verdict(verdict, level=5) for verdict in example_verdicts]
    mapped_verdicts_two = [map_verdict(verdict, level=2) for verdict in example_verdicts]

    mapped_verdicts_seven, mapped_verdicts_five, mapped_verdicts_two


    true_labels_lower = [label.lower() for label in true_labels]
    mapped_true_labels_lower = {level: [map_verdict(verdict, level=level) for verdict in true_labels_lower] for level in [7, 5, 2]}[2]
    predicted_results_lower = [result.lower() for result in predicted_results]
    report = classification_report(mapped_true_labels_lower, predicted_results_lower)
    print(report)

if __name__ == "__main__":
    main()