import json
import os
import sys
import re

prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)


from prompt.chatgpt_prompt import CHATGPT_CODE_INTERPRETER_SYSTEM_PROMPT
from .JuypyterClient import JupyterNotebook


class BaseCodeInterpreter:
    def __init__(self):
        self.dialog = [
            {
                "role": "system",
                "content": CHATGPT_CODE_INTERPRETER_SYSTEM_PROMPT,
            },
        ]

        self.nb = JupyterNotebook()

    @staticmethod
    def extract_code_blocks(text: str):
        pattern = r"```python\n(.*?)```"  # Only match blocks with 'python\n' syntax highlighting tag
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    def execute_code_and_return_output(self, code_str: str):
        outputs, error_flag = self.nb.add_and_run(code_str)
        return outputs, error_flag

    def close(self):
        if self.nb and hasattr(self.nb, "close"):
            try:
                self.nb.close()
            except Exception as e:
                # Maybe log the exception for debugging
                pass

    def __del__(self):
        self.close()