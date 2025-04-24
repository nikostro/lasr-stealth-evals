from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from inspect_ai.log import read_eval_log

from lasr_stealth_evals.collusion.config import WANDB_API_KEY

wandb.login(key=WANDB_API_KEY)


def analyze_strategy_hint_patterns(log_path: Path):
    """Analyze strategy vs hint patterns from an evaluation log file."""
    # Read the evaluation log
    log = read_eval_log(str(log_path))

    # Extract strategy and hint information from samples
    data = []
    for sample in log.samples:  # type: ignore
        metadata = sample.metadata
        strategies: Optional[List[Dict[str, Any]]] = metadata.get("strategies")
        if strategies is None:
            strategies = []

        hint = metadata.get("hint", {})

        # Get strategy names
        strategy_names = [s["name"] for s in strategies] if strategies else ["none"]

        # Get hint name
        hint_name = hint.get("name", "unknown")

        # Get the judge's score (C for correct/collusion, I for incorrect/no collusion)
        judge_score = sample.scores["judge"].value if sample.scores["judge"] else None  # type: ignore
        outcome = judge_score == "C"  # True if judge marked as collusion (C)

        data.append(
            {
                "strategy": strategy_names[0] if len(strategy_names) == 1 else "multiple",
                "hint": hint_name,
                "outcome": int(outcome),
                "score": judge_score,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create a pivot table of strategy vs hint
    pivot = pd.pivot_table(df, values="outcome", index="strategy", columns="hint", aggfunc="mean")

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5, vmin=0, vmax=1)
    plt.title("Strategy vs Hint Collusion Success Rate")
    plt.xlabel("Hint")
    plt.ylabel("Strategy")
    plt.tight_layout()

    # Save the plot
    output_path = log_path.parent / f"{log_path.stem}_analysis.png"
    plt.savefig(output_path)
    print(f"Analysis saved to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nCollusion success rate by strategy:")
    print(df.groupby("strategy")["outcome"].mean().sort_values(ascending=False))  # type: ignore
    print("\nCollusion success rate by hint:")
    print(df.groupby("hint")["outcome"].mean().sort_values(ascending=False))  # type: ignore

    # Print score distribution
    print("\nScore distribution:")
    print(df["score"].value_counts())


if __name__ == "__main__":
    # Get the most recent log file
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_files = sorted(log_dir.glob("*.eval"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not log_files:
        print("No evaluation log files found")
        exit(1)

    most_recent_log = log_files[0]
    print(f"Analyzing most recent log: {most_recent_log}")
    analyze_strategy_hint_patterns(most_recent_log)
