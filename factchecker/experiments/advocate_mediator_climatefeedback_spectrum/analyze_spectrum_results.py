"""
Re-run analysis on an existing Spectrum results CSV (entry point from experiments).

Business logic in factchecker.utils.spectrum_analysis.
Canonical CLI: python -m factchecker.tools.analyze_spectrum_results

Usage:
  python -m factchecker.experiments.advocate_mediator_climatefeedback_spectrum.analyze_spectrum_results
  python -m factchecker.experiments.advocate_mediator_climatefeedback_spectrum.analyze_spectrum_results --results path/to/spectrum_claims_results_*.csv
"""

import argparse
import sys
from pathlib import Path

from factchecker.utils.spectrum_analysis import (
    find_latest_spectrum_results_csv,
    run_spectrum_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze existing Spectrum results CSV (weighted metrics, optional classification report)."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to spectrum_claims_results_*.csv. If omitted, uses most recent in experiments/results/.",
    )
    args = parser.parse_args()

    if args.results:
        path = Path(args.results)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        csv_path = str(path)
    else:
        latest = find_latest_spectrum_results_csv()
        if latest is None:
            print(
                "Error: No spectrum_claims_results_*.csv found. Pass --results path/to/file.csv",
                file=sys.stderr,
            )
            sys.exit(1)
        csv_path = str(latest)

    result = run_spectrum_analysis(csv_path)

    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {result.n_rows} rows from {csv_path}\n")

    if result.classification_report:
        print("Classification Metrics (primary verdict vs true):")
        print(result.classification_report)

    if result.weighted_metrics_text:
        print(result.weighted_metrics_text)


if __name__ == "__main__":
    main()
