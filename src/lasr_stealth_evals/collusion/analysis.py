from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from inspect_ai.log import read_eval_log

import wandb
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


def analyze_agent_model_patterns(log_paths: List[str]):
    """Analyze agent model patterns from a list of evaluation log files."""
    # Read the evaluation logs
    logs = [(log_path.split("/")[-1], read_eval_log(log_path)) for log_path in log_paths]
    domains = ["social_media", "coding"]
    # Extract agent model information from samples
    data = {}
    for model_name, log in logs:
        processed_model_name = model_name.split(".")[0]
        for domain in domains:
            if (processed_model_name, domain) not in data:
                data[(processed_model_name, domain)] = []
            for sample in log.samples:  # type: ignore
                metadata = sample.metadata
                task_name = metadata.get("task_name", "")
                if not task_name.startswith(domain):
                    continue
                data[(processed_model_name, domain)].append(
                    sample.scores["judge"].value if "judge" in sample.scores else 0  # type: ignore
                )

    # Create plot comparing model performance across domains
    plt.figure(figsize=(12, 6))

    # Calculate success rates and standard errors for each model/domain
    success_rates = {}
    std_errors = {}
    for (model, domain), scores in data.items():
        # Convert scores to binary (1 for "C", 0 for others)
        binary_scores = [1 if s == "C" else 0 for s in scores]
        success_rate = np.mean(binary_scores) if scores else 0
        # Calculate standard error
        std_error = np.std(binary_scores, ddof=1) / np.sqrt(len(binary_scores)) if scores else 0
        success_rates[(model, domain)] = success_rate
        std_errors[(model, domain)] = std_error

    # Prepare data for plotting
    models = sorted(set(k[0] for k in success_rates.keys()))
    x = np.arange(len(models))
    width = 0.35

    # Create grouped bars for each domain
    for i, domain in enumerate(domains):
        rates = [success_rates.get((model, domain), 0) for model in models]
        errors = [std_errors.get((model, domain), 0) for model in models]
        plt.bar(x + i * width, rates, width, label=domain, yerr=errors, capsize=5)

    plt.xlabel("Models")
    plt.ylabel("Acceptance Rate (vs. GPT-4.1 Monitor)")
    plt.title("Agent Model Performance by Domain")
    plt.xticks(x + width / 2, models, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = Path(log_paths[0]).parent / "model_domain_analysis.png"
    plt.savefig(output_path)
    print(f"Model domain analysis saved to {output_path}")


def analyze_cot_patterns(log_paths: List[str]):
    """Analyze and visualize the impact of chain-of-thought (cot) on different strategies."""
    logs = [read_eval_log(log_path) for log_path in log_paths]
    print(len(logs))
    data = {}
    for log in logs:
        k = "cot" if "cot" in log.eval.task_args and log.eval.task_args["cot"] else "no_cot"
        assert len(log.samples) > 0, "No samples found in log"  # type: ignore
        strat_type = log.samples[0].metadata.get("strategies", "")  # type: ignore
        strat_type = "none" if len(strat_type) == 0 else "all"
        if (strat_type, k) not in data:
            data[(strat_type, k)] = []
        for sample in log.samples:  # type: ignore
            data[(strat_type, k)].append(sample.scores["judge"].value if "judge" in sample.scores else 0)  # type: ignore

    print(data.keys())
    # Calculate success rates and standard errors for each strategy/cot combination
    success_rates = {}
    std_errors = {}
    for (strategy, cot_type), scores in data.items():
        # Convert scores to binary (1 for "C", 0 for others)
        binary_scores = [1 if s == "C" else 0 for s in scores]
        success_rate = np.mean(binary_scores) if scores else 0
        # Calculate standard error
        std_error = np.std(binary_scores, ddof=1) / np.sqrt(len(binary_scores)) if scores else 0
        success_rates[(strategy, cot_type)] = success_rate
        std_errors[(strategy, cot_type)] = std_error

    # Prepare data for plotting
    strategies = sorted(set(k[0] for k in success_rates.keys()))
    x = np.arange(len(strategies))
    width = 0.35

    # Create grouped bars for each strategy
    plt.figure(figsize=(12, 6))

    # Define colors for cot and no_cot
    colors = {"cot": "#2ecc71", "no_cot": "#e74c3c"}  # Green for cot, Red for no_cot

    # Plot bars for each strategy
    for i, strategy in enumerate(strategies):
        for j, cot_type in enumerate(["cot", "no_cot"]):
            rate = success_rates.get((strategy, cot_type), 0)
            error = std_errors.get((strategy, cot_type), 0)
            plt.bar(
                x[i] + (j - 0.5) * width,
                rate,
                width,
                color=colors[cot_type],
                label=cot_type if i == 0 else "",
                yerr=error,
                capsize=5,
            )

    plt.xlabel("Strategy")
    plt.ylabel("Collusion Success Rate")
    plt.title("Impact of Chain-of-Thought on Strategy Performance")
    plt.xticks(x, strategies, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = Path(log_paths[0]).parent / "cot_strategy_analysis.png"
    plt.savefig(output_path)
    print(f"COT strategy analysis saved to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        for cot_type in ["cot", "no_cot"]:
            rate = success_rates.get((strategy, cot_type), 0)
            error = std_errors.get((strategy, cot_type), 0)
            print(f"{cot_type}: {rate:.3f} ± {error:.3f}")


if __name__ == "__main__":
    # Get the most recent log file
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_files = sorted(log_dir.glob("*.eval"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not log_files:
        print("No evaluation log files found")
        exit(1)

    most_recent_log = log_files[0]
    print(f"Analyzing most recent log: {most_recent_log}")
    # analyze_strategy_hint_patterns(most_recent_log)
    # analyze_agent_model_patterns(
    #     [str(log_dir / "claude-3.7-sonnet.eval"), str(log_dir / "gpt-4.1.eval"), str(log_dir / "deepseek-chat.eval")]
    # )
    analyze_cot_patterns(
        [
            str(log_dir / "no_cot_all.eval"),
            str(log_dir / "cot_all.eval"),
            str(log_dir / "cot_no_strat.eval"),
            str(log_dir / "no_cot_no_strat.eval"),
        ]
    )
