# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %% VISUALIZATION 1: Grouped Bar Chart
# Convert boolean inputs to string labels for visualization
def get_label(value: bool, prefix: str) -> str:
    return f"{'True' if value else 'False'}"


# Assuming input_data is your JSON object with format:
# [{"model": str, "hint": bool, "context": bool, "accuracy": float}, ...]
def create_visualization(input_data: list[dict[str, bool | float]]):
    # Convert the data to DataFrame
    df = pd.DataFrame(input_data)

    # Convert boolean values to readable labels
    df["Hint"] = df["hint"].apply(lambda x: get_label(x, "Hint"))
    df["Context"] = df["context"].apply(lambda x: get_label(x, "Context"))
    df["Model"] = df["model"]

    # Create the visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    ax = sns.catplot(
        data=df, kind="bar", x="Context", y="accuracy", hue="Hint", col="Model", palette="viridis", height=5, aspect=0.8
    )

    ax.set_titles("{col_name}")
    ax.set_axis_labels("Context", "Accuracy")
    ax.add_legend(title="Hint")
    plt.tight_layout()
    plt.savefig("grouped_bar_chart.png", dpi=300)

    return ax


# Example usage:
input_data = [
    {"model": "Model A", "hint": True, "context": True, "accuracy": 0.78},
    {"model": "Model A", "hint": False, "context": True, "accuracy": 0.82},
    {"model": "Model B", "hint": True, "context": True, "accuracy": 0.78},
    {"model": "Model B", "hint": False, "context": True, "accuracy": 0.82},
]
create_visualization(input_data)

# %%
