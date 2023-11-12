import os
import argparse
from rich.console import Console
from rich.table import Table

console = Console()

DS1000_ALL_TASKS = [
    "Numpy",
    "Pandas",
    "Tensorflow",
    "Pytorch",
    "Sklearn",
    "Matplotlib",
    "Scipy",
]
DS1000_ALL_TYPES = ["Surface", "Origin", "Semantic", "Difficult-Rewrite"]


def evaluate_log(log: list) -> str:
    correct = 0
    for i in log:
        if "True" in i:
            correct += 1

    return f"{str(round(correct / len(log), 2))} ({correct}/{len(log)})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process memory directory path")
    parser.add_argument(
        "--memory_dir",
        help="Path to the memory directory",
        default="./memory/SelfDocAgent/gpt-4",
        type=str,
    )
    args = parser.parse_args()

    logs = [i for i in os.listdir(args.memory_dir) if i.endswith(".json")]
    logs = [log for log in logs if any(task in log for task in DS1000_ALL_TASKS)]
    console.print(f"{args.memory_dir} : [bold green]{len(logs)}[/bold green]")

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task", style="dim")
    table.add_column("Type")
    table.add_column("Score")

    # Populate the table
    for task in DS1000_ALL_TASKS:
        for type in DS1000_ALL_TYPES:
            relevant_logs = [log for log in logs if task in log and type in log]
            if len(relevant_logs) > 0:
                score = evaluate_log(relevant_logs) if relevant_logs else "N/A"
                table.add_row(task, type, score)

    # Print the table
    console.print(table)
