import re
from typing import List, Dict
import logging
import logging.handlers

from rich.logging import RichHandler
from rich.console import Console
import sys

import os
from retrying import retry
import requests
import openai
import numpy as np


def remove_string(s):
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:.*LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64\n"
    return re.sub(pattern, "", s)


def dialog_to_string(
    dialog: List[Dict], pretty: bool = False, include_system_prompt: bool = False
):
    return_string = ""
    for content_dict in dialog:
        role = content_dict["role"]
        content = content_dict["content"]
        if (role.lower() == "system") and (not include_system_prompt):
            continue
        if pretty and (role.lower() == "user"):
            return_string += f"👤  {role.upper()} :\n"
        elif pretty and (role.lower() == "assistant"):
            return_string += f"🤖  {role.upper()} :\n"
        else:
            return_string += f"{role.upper()} :\n"
        return_string += f"{content}\n"
    return return_string


def extract_code_block(text):
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def set_logger(log_path: str = "application.log") -> logging.Logger:
    RICH_FORMAT = "%(message)s"  # Let Rich handle the formatting
    FILE_HANDLER_FORMAT = "[%(asctime)s]\\t%(levelname)s\\t[%(filename)s:%(funcName)s:%(lineno)s]\\t>> %(message)s"

    logger = logging.getLogger("rich")
    logger.setLevel(logging.INFO)

    # Clear existing handlers from the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # RichHandler
    rich_handler = RichHandler(
        rich_tracebacks=True, markup=True
    )  # Ensure markup is True
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(logging.Formatter(RICH_FORMAT))
    logger.addHandler(rich_handler)

    # FileHandler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)

    return logger


@retry(
    stop_max_attempt_number=7,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
def get_embedding_ada(text: str) -> np.array:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {"input": text, "model": "text-embedding-ada-002"}

    response = requests.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=payload
    )

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}\n\t{response.text}"
        )

    return np.array(response.json()["data"][0]["embedding"])
