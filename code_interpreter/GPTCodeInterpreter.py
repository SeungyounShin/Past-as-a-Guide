import os, sys

# Get the path from environment variable
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)

from .JuypyterClient import JupyterNotebook
from .BaseCodeInterpreter import BaseCodeInterpreter
from rich.console import Console
from rich.markdown import Markdown
from utils.utils import *

import openai
from retrying import retry
from dotenv import load_dotenv

load_dotenv()

from prompt.chatgpt_prompt import *


class GPTCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.dialog = [
            {
                "role": "system",
                "content": CHATGPT_CODE_INTERPRETER_SYSTEM_PROMPT + "\n",
            },
        ]

        self.response = None

        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.nb = JupyterNotebook()
        PRE_EXEC_CODE_OUT = self.nb.add_and_run(PRE_EXEC_CODE)

        self.console = Console()  # for printing output

    def get_response_content(self):
        if self.response:
            return self.response["choices"][0]["message"]["content"]
        else:
            return None

    def clean_the_dialog_one_step(self, question):
        question_idx = 0
        for idx, item in enumerate(self.dialog):
            if item["content"] == question:
                question_idx = idx

        filtered_dialog = self.dialog[question_idx:]

        user_qinit_dict = filtered_dialog[0]
        answer_fuse_str = "\n".join(
            [i["content"].strip() for i in filtered_dialog[1::2]]
        )

        self.dialog = self.dialog[:question_idx] + [
            {"role": "user", "content": user_qinit_dict["content"]},
            {"role": "assistant", "content": answer_fuse_str},
        ]

    def close(self):
        if self.nb and hasattr(self.nb, "close"):
            try:
                self.nb.close()
            except Exception as e:
                # Maybe log the exception for debugging
                pass

    def __del__(self):
        self.close()

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def ChatCompletion(self, temperature: float = 0.1, top_p: float = 1.0):
        dialog_stream = openai.ChatCompletion.create(
            model=self.model,
            messages=self.dialog,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        stop_string1 = "```python"
        stop_string2 = "```"
        stop_condition_met1, stop_condition_met2 = False, False
        in_code_block = False  # Flag to denote if we are inside a code block
        buffer = ""

        for chunk in dialog_stream:
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                buffer += content  # add received content to the buffer
                yield (
                    content,
                    (stop_condition_met1, stop_condition_met2),
                )  # yield received content

                if stop_string1 in buffer:
                    stop_condition_met1 = True

                elif stop_condition_met1 and (stop_string2 in buffer):
                    stop_condition_met2 = True

                # You may want to clear or trim the buffer to manage memory usage.
                if len(buffer) > 100:
                    buffer = buffer[-100:]  # Keep the last 1000 characters

                # If the stop condition is met, break out of the loop
                if stop_condition_met2:
                    break

    def chat(
        self,
        user_message: str,
        MAX_TRY: int = 6,
        temperature: float = 0.1,
        top_p: float = 1.0,
        VERBOSE: bool = True,
    ):
        if VERBOSE:
            self.console.print(Markdown(f"👤 User : **{user_message}**\n"))
            self.console.print(Markdown(f"🤖 Assistant :\n"))
        self.dialog.append({"role": "user", "content": user_message})

        # interactively and interatively code generation and refinement
        for i in range(MAX_TRY):
            generated_text_local = ""
            for char, cond in self.ChatCompletion(temperature=temperature, top_p=top_p):
                generated_text_local += char

                if VERBOSE:
                    if cond[0]:
                        self.console.print(char, style="code", end="")
                    elif cond[-1]:
                        self.console.print(char, end="")
                    else:
                        self.console.print(char, end="")

            # Get code block
            code_blocks = self.extract_code_blocks(generated_text_local)

            if code_blocks:
                code_output, error_flag = self.execute_code_and_return_output(
                    code_blocks[0]
                )

                response_content = (
                    f"{generated_text_local}\nExecution Result:\n{code_output}\n"
                )
                self.console.print(
                    f"```Execution Result:\n{code_output}\n```\n", style="code", end=""
                )
                self.dialog.append({"role": "assistant", "content": response_content})
                self.dialog.append({"role": "user", "content": CHATGPT_FEEDBACK_PROMPT})

            else:
                if "<done>" in generated_text_local:
                    generated_text_local = generated_text_local.split("<done>")[
                        0
                    ].strip()

                self.dialog.append(
                    {"role": "assistant", "content": generated_text_local}
                )
                break

        # make all steps looks like an one answer
        self.clean_the_dialog_one_step(question=user_message)

        return self.dialog


if __name__ == "__main__":
    # "plot 30days BTC price? using yfinance all so print current price"
    gpt_interpreter = GPTCodeInterpreter()

    answer = gpt_interpreter.chat("what is 77th fibonacci number?")
    gpt_interpreter.close()
