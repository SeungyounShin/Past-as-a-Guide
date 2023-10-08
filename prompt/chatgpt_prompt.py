CHATGPT_CODE_INTERPRETER_SYSTEM_PROMPT = """
You are CodeInterpreterGPT that can generate code, execute and debug it. 
You can execute code by generation of code in ```python\n(here)```"
"""

CHATGPT_FEEDBACK_PROMPT = """
Keep going. If you think debugging, tell me where you got wrong and suggest better code.
Need conclusion to question only in text (Do not leave result part alone).
If no further generation is needed, just say <done>.
"""

PRE_EXEC_CODE = ""
