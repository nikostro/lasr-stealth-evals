import matplotlib.pyplot as plt
from pathlib import Path
from lasr_stealth_evals.insider_trading.env import Trader

from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Dict, List
from lasr_stealth_evals.insider_trading.config import OPENAI_API_KEY
from autogen_core.models import ChatCompletionClient


# Initialisations


def init_models() -> Dict[str, List[ChatCompletionClient]]:
    return {
        "openai": [
            OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
            )
        ]
    }


# Plotting
def plot_trader_holdings(trader: Trader, plot_dir: Path, filename: str = "trader_holdings.png"):
    """
    Create and save a bar chart of the trader's holdings.

    Args:
        trader: The Trader object to plot holdings for
        plot_dir: Directory to save the plot in
        filename: Name of the file to save the plot as
    """
    # Create plots directory if it doesn't exist
    plot_dir.mkdir(exist_ok=True)

    # Create bar chart
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting
    assets = list(trader.assets.keys())
    values = list(trader.assets.values())

    # Add cash as the first bar
    assets.insert(0, "Cash")
    values.insert(0, trader.cash)

    # Create bar chart
    bars = plt.bar(assets, values)

    # Customize the chart
    plt.title(Path(filename).stem.replace("_", " ").title())
    plt.xlabel("Asset")
    plt.ylabel("Amount")
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:,.0f}", ha="center", va="bottom")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_dir / filename)
    plt.close()
