#!/usr/bin/env python
"""
FlowGSP Experiments CLI

A command-line interface for running reproducible experiments
organized by paper.

Usage:
    # List all papers and experiments
    python -m experiments.run --list

    # Run a specific experiment
    python -m experiments.run --paper paper_test_covar --experiment exp_asymmetry_erdos_graph
"""

import argparse
import importlib
import sys
from pathlib import Path
import numpy as np

# Get the experiments directory
EXPERIMENTS_DIR = Path(__file__).parent.resolve()


def discover_papers() -> dict:
    """
    Discover all papers and their experiments.

    Returns
    -------
    dict
        Dictionary mapping paper names to lists of experiment names.
    """
    papers = {}

    for item in EXPERIMENTS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("paper"):
            paper_name = item.name
            experiments = []

            for exp_item in item.iterdir():
                if exp_item.is_dir() and exp_item.name.startswith("exp"):
                    # Check if it has code/experiment.py
                    exp_script = exp_item / "code" / "experiment.py"
                    if exp_script.exists():
                        experiments.append(exp_item.name)

            if experiments:
                papers[paper_name] = sorted(experiments)

    return papers


def list_papers_and_experiments() -> None:
    """Print a list of all available papers and experiments."""
    papers = discover_papers()

    if not papers:
        print("No papers or experiments found.")
        print("\nTo add experiments, create directories following this structure:")
        print("  experiments/paper_<name>/exp_<name>/code/experiment.py")
        return

    print("\n" + "=" * 60)
    print("FlowGSP Experiments")
    print("=" * 60)

    # Paper numbering based on readme order
    paper_number = []
    for paper_name in papers.keys():
        paper_readme = EXPERIMENTS_DIR / paper_name / "README.md"
        if paper_readme.exists():
            with open(paper_readme, "r") as f:
                title = f.readlines()[0]
                if title.startswith("# "):
                    numbering = int(title.split(" ")[2][:-1])  # Numbering
        paper_number.append(numbering)
    sorted_papers = np.array(list(papers.keys()))[np.argsort(paper_number)]
    for paper_name in sorted_papers:
        experiments = papers[paper_name]
        paper_readme = EXPERIMENTS_DIR / paper_name / "README.md"

        # Try to get paper title from README
        title = paper_name
        if paper_readme.exists():
            with open(paper_readme, "r") as f:
                for line in f:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

        print(f"\n📄 {title}")
        print(f"   Directory: {paper_name}/")

        for exp_name in experiments:
            exp_readme = EXPERIMENTS_DIR / paper_name / exp_name / "README.md"

            # Try to get experiment title from README
            exp_title = exp_name
            if exp_readme.exists():
                with open(exp_readme, "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            exp_title = line[2:].strip()
                            break

            print(f"   └── 🔬 {exp_title}")
            print(
                f"       Run: python -m experiments.run --paper {paper_name} --experiment {exp_name}"
            )

    print("\n" + "=" * 60)
    print("For more details, see experiments/PAPERS_AND_EXPERIMENTS.md")
    print("=" * 60 + "\n")


def run_experiment(paper: str, experiment: str, verbose: bool = True) -> int:
    """
    Run a specific experiment.

    Parameters
    ----------
    paper : str
        Name of the paper directory.
    experiment : str
        Name of the experiment directory.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    papers = discover_papers()

    # Validate paper
    if paper not in papers:
        print(f"Error: Paper '{paper}' not found.")
        print(f"Available papers: {', '.join(sorted(papers.keys()))}")
        return 1

    # Validate experiment
    if experiment not in papers[paper]:
        print(f"Error: No experiment '{experiment}' found in paper '{paper}'.")
        print(f"Available experiments: {', '.join(papers[paper])}")
        return 1

    # Import and run the experiment
    try:
        # Construct module path
        module_path = f"experiments.{paper}.{experiment}.code.experiment"

        if verbose:
            print(f"\nRunning experiment: {paper}/{experiment}")
            print(f"Module: {module_path}\n")

        # Import the experiment module
        experiment_module = importlib.import_module(module_path)

        # Run the experiment
        if hasattr(experiment_module, "run"):
            experiment_module.run(verbose=verbose)
            return 0
        else:
            print("Error: Experiment module does not have a 'run' function.")
            return 1

    except ImportError as e:
        print(f"Error importing experiment module: {e}")
        return 1
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="FlowGSP Experiments CLI - Run reproducible experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all papers and experiments:
    python -m experiments.run --list

  Run a specific experiment:
    python -m experiments.run --paper paper_test_covar --experiment exp_asymmetry_erdos_graph

For more information, see experiments/README.md
        """,
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available papers and experiments",
    )

    parser.add_argument(
        "--paper",
        "-p",
        type=str,
        help="Name of the paper directory (e.g., paper_test_covar)",
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Name of the experiment directory (e.g., exp_asymmetry_erdos_graph)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_papers_and_experiments()
        return 0

    # Handle running an experiment
    if args.paper and args.experiment:
        return run_experiment(args.paper, args.experiment, verbose=not args.quiet)

    # If neither --list nor both --paper and --experiment are provided
    if args.paper and not args.experiment:
        print("Error: --experiment is required when --paper is specified.")
        return 1

    if args.experiment and not args.paper:
        print("Error: --paper is required when --experiment is specified.")
        return 1

    # No arguments provided, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
