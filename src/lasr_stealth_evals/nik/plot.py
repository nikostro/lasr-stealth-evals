# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_visualization(
    input_data: list[dict[str, dict[str, str | bool | float]]],
    metric: str = "precision",
    output_dir: str = "plots1",
):
    df = pd.DataFrame(
        [
            {
                "model": item["config"]["model"],
                "hint": item["config"]["hint"],
                "context": item["config"]["context"],
                metric: item["results"][metric],
            }
            for item in input_data
        ]
    )

    # Convert boolean values to readable labels
    df["Hint"] = df["hint"].astype(str)
    df["Context"] = df["context"].astype(str)
    df["Model"] = df["model"]

    # Create the visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Ensure the plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Custom color palette: using a more visually distinct and appealing scheme
    custom_palette = {"False": "#1f77b4", "True": "#ff7f0e"}  # Blue for True, Orange for False

    ax = sns.catplot(
        data=df,
        kind="bar",
        x="Context",
        y=metric,
        hue="Hint",
        col="Model",
        palette=custom_palette,
        height=5,
        aspect=0.8,
        order=["False", "True"],  # Set the order of x-axis categories: False first, then True
        hue_order=["False", "True"],  # Set the order of hint values in the legend
    )

    ax.set_titles("{col_name}")
    ax.set_axis_labels("Context", metric.capitalize())

    # Move the legend to the top left corner of left plot only
    ax.figure.legends.clear()  # Clear existing legends
    ax.axes.flat[0].legend().remove()  # Remove legend from right plot
    ax.axes.flat[1].legend(title="Hint", loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/chart_{metric}.png", dpi=300)

    return ax


# Load data from metrics.json
with open("plots/metrics1.json", "r") as f:
    data = json.load(f)


# Plot accuracy by default
for metric in ["precision", "recall", "avoidance", "accuracy"]:
    create_visualization(data, metric)


# %%
