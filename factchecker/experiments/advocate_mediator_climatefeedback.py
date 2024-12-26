from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy
from factchecker.prompts.advocate_mediator_prompts import advocate_primer, arbitrator_primer
import pandas as pd
from sklearn.metrics import classification_report
from llama_index.core import Settings

Settings.chunk_size=150
Settings.chunk_overlap=20

from datetime import datetime
import os

def main():
    def map_verdict(verdict, level=2):
        verdict = verdict.strip().lower().replace(" ", "_")

        if level == 7:
            mapping = {
                "incorrect": "incorrect",
                "inaccurate": "inaccurate",
                "imprecise": "imprecise",
                "misleading": "misleading",
                "flawed_reasoning": "flawed_reasoning",
                "lacks_context": "lacks_context",
                "unsupported": "unsupported",
                "correct_but": "correct_but",
                "mostly_correct": "mostly_correct",
                "mostly_accurate": "mostly_accurate",
                "accurate": "accurate",
                "correct": "correct"
            }
        elif level == 5:
            mapping = {
                "incorrect": "incorrect",
                "inaccurate": "incorrect",
                "imprecise": "imprecise",
                "misleading": "misleading",
                "flawed_reasoning": "flawed_reasoning",
                "lacks_context": "unsupported",
                "unsupported": "unsupported",
                "correct_but": "mostly_correct",
                "mostly_correct": "mostly_correct",
                "mostly_accurate": "correct",
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
                print(f"Unmapped verdict: {verdict}")  # Add this line to identify unmapped verdicts
                return "unknown"

        mapped_verdict = mapping.get(verdict, "unknown")
        if mapped_verdict == "unknown":
            print(f"Unmapped verdict: {verdict}")  # Add this line to identify unmapped verdicts
        return mapped_verdict

    indexer_options_list = [
        {'source_directory': 'data'},  # Indexer options for advocate 1
        # {'source_directory': 'data/source2'},  # Indexer options for advocate 2
        # {'source_directory': 'data/source3'}   # Indexer options for advocate 3
    ]

    retriever_options_list = [
        {'retrieval_options': {'similarity_top_k': 8, 'min_score': 0.5}, 'indexer_options': indexer_options_list[0]},  # Options for advocate 1
        # {'retrieval_options': {'similarity_top_k': 8}, 'indexer_options': indexer_options_list[1]},  # Options for advocate 2
        # {'retrieval_options': {'similarity_top_k': 8}, 'indexer_options': indexer_options_list[2]}   # Options for advocate 3
    ]

    advocate_options = {
        'max_evidences': 10,        # Add other advocate step options here
    }
    mediator_options = {
        # Add other mediator step options here
    }

    # Initialize the advocate-mediator strategy with the options and prompts
    strategy = AdvocateMediatorStrategy(indexer_options_list, retriever_options_list, advocate_options, mediator_options, advocate_primer, arbitrator_primer)

    # Load the claims from the CSV file
    # Load the claims from the CSV file
    claims_df = pd.read_csv('datasets/Combined_Overview_Climate_Feedback_Claims.csv')

    # Debug: Print the number of rows in the dataset
    print(f"Total number of claims: {len(claims_df)}")

    # Filter to include only correct claims
    correct_claims_df = claims_df[claims_df['Climate Feedback'].str.lower().isin(['correct', 'mostly correct', 'accurate', 'mostly accurate', 'correct but'])]

    # Debug: Print the number of correct claims
    print(f"Number of correct claims: {len(correct_claims_df)}")

    # Calculate the number of correct claims needed
    num_claims_to_test = 100  # Example value, can be changed as needed
    num_correct_claims_needed = int(num_claims_to_test * 0.3)

    # Ensure we have enough correct claims
    if len(correct_claims_df) < num_correct_claims_needed:
        num_correct_claims_needed = len(correct_claims_df)

    # Sample the correct claims
    sampled_correct_claims_df = correct_claims_df.sample(n=num_correct_claims_needed, random_state=42)

    # Sample the remaining claims from the original dataset
    remaining_claims_needed = num_claims_to_test - num_correct_claims_needed
    remaining_claims_df = claims_df.drop(sampled_correct_claims_df.index)

    # Debug: Print the number of remaining claims
    print(f"Number of remaining claims: {len(remaining_claims_df)}")

    # Ensure we do not sample more than available in remaining_claims_df
    if len(remaining_claims_df) < remaining_claims_needed:
        remaining_claims_needed = len(remaining_claims_df)

    sampled_remaining_claims_df = remaining_claims_df.sample(n=remaining_claims_needed, random_state=42)

    # Combine the sampled correct claims with the remaining sampled claims
    sampled_claims_df = pd.concat([sampled_correct_claims_df, sampled_remaining_claims_df]).reset_index(drop=True)

    # Prepare lists to store true labels, predicted results, and mediator reasoning
    true_labels = []
    predicted_results = []
    mediator_reasonings = []

    # Initialize advocate-related lists after the first evaluation to determine the number of advocates
    advocate_evidences = []
    advocate_verdicts = []
    advocate_reasonings = []

    # Iterate over the sampled claims
    for index, row in sampled_claims_df.iterrows():
        # Extract the claim and the true label
        claim = row['Claim']
        true_label = row['Climate Feedback']

        # Evaluate the claim using the strategy
        final_verdict, verdicts, reasonings = strategy.evaluate_claim(claim)

        # Determine the number of advocates dynamically
        if not advocate_evidences:
            num_advocates = len(verdicts)
            advocate_evidences = [[] for _ in range(num_advocates)]
            advocate_verdicts = [[] for _ in range(num_advocates)]
            advocate_reasonings = [[] for _ in range(num_advocates)]

        # Append the results for later evaluation
        true_labels.append(true_label)
        predicted_results.append(final_verdict)
        mediator_reasonings.append(reasonings[-1])  # Assuming the last reasoning is from the mediator

        for i in range(num_advocates):
            advocate_evidences[i].append(verdicts[i])
            advocate_verdicts[i].append(verdicts[i])
            advocate_reasonings[i].append(reasonings[i])

        # Print the results for each claim
        print(f"True Label: {true_label}")
        mapped_true_label = map_verdict(true_label)
        print(f"Mapped True Label: {mapped_true_label}")
        print(f"Claim: {claim}")
        print(f"Final Verdict: {final_verdict}")
        print(f"Advocate Verdicts: {verdicts}")
        print(f"Reasoning: {reasonings}")

    # Create a dictionary to store the results
    results_dict = {
        'Claim': sampled_claims_df['Claim'],
        'True Label': true_labels,
        'Mapped True Label': [map_verdict(label) for label in true_labels],
        'Predicted Verdict': predicted_results,
        'Mediator Reasoning': mediator_reasonings
    }

    # Add advocate columns to the results dictionary
    for i in range(num_advocates):
        results_dict[f'Advocate {i+1} Evidence'] = advocate_evidences[i]
        results_dict[f'Advocate {i+1} Verdict'] = advocate_verdicts[i]
        results_dict[f'Advocate {i+1} Reasoning'] = advocate_reasonings[i]

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results_dict)

    # Add a timestamp to the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"experiments/results/claims_results_{timestamp}.csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)

    # Save the results to a CSV file
    results_df.to_csv(results_filename, index=False)

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