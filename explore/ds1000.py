import sys, os, json
from agent.self_doc_agent import SelfDocAgent


def automoatic_load_ds1000():
    root_path = os.path.abspath("./")
    DS1000_abs_path = os.path.join(root_path, "explore/DS-1000")
    sys.path.append(DS1000_abs_path)
    from ds1000 import DS1000Dataset

    ds_data = DS1000Dataset(
        os.path.join(DS1000_abs_path, "ds1000_data")
    )  # loads all questions into RAM
    return ds_data


def format_ds1000_question(question: str) -> str:
    SOLUTION_TEXT = "Fill Solution Code"
    # Replace <code> and </code> with appropriate markdown
    formatted_question = question.replace("<code>", "```python").replace(
        "</code>", "```"
    )

    # Format BEGIN SOLUTION and END SOLUTION segment
    start_solution_index = formatted_question.find("BEGIN SOLUTION")
    end_solution_index = formatted_question.find("END SOLUTION")

    # Extract the solution code
    solution_code = formatted_question[
        start_solution_index + len("BEGIN SOLUTION") : end_solution_index
    ].strip()

    formatted_question = (
        formatted_question[:start_solution_index]
        + f"\n<{SOLUTION_TEXT}>\n"
        + formatted_question[end_solution_index + len("END SOLUTION") :]
    )
    formatted_question = formatted_question.replace(
        f"```\n\n<{SOLUTION_TEXT}>\n\n```python",
        f"<{SOLUTION_TEXT}>",
    )

    return formatted_question


if __name__ == "__main__":
    # load ds1000
    # ds_data = automoatic_load_ds1000()
    # formatted_question = format_ds1000_question(ds_data["Pandas"][0]["prompt"])
    formatted_question = (
        "Can you plot the volatility smiles for each expiration of Tesla Stock?"
    )

    agent = SelfDocAgent()
    agent.step(instruction=formatted_question)

    agent.close()
