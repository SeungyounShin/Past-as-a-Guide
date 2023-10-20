from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from prompt.chatgpt_explore_prompt import exploration_instruction
from utils.utils import *

import os, sys, json
from typing import List


class MemorySegment:
    def __init__(self):
        pass


class MemoryBank:
    def __init__(self, model: str = "gpt-4", retrieval_model: str = ""):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)

        self.memory_dir_path = os.path.join(parent_directory, f"memory/{model}")

        if not os.path.exists(self.memory_dir_path):
            os.makedirs(self.memory_dir_path)

        self.memory_segment_files = [
            os.path.join(memory_dir_path, i) for i in os.listdir(self.memory_dir_path)
        ]
        self.set_memory_on_disk()

    def set_memory_on_disk(self):
        pass

    def encode(self, dialog: List, retrieval_k: int = 3):
        """
        When Agent Make Trajectories of (Instruction, Answer) (it can be multihop)
        Like human (not saving raw dialog but encode then save)
        thie module encode memory
        """
        # step 1 : Modify previous document If needed
        # print(dialog_to_string(dialog=dialog))
        exit()

        # step 2 : Retrieve the k similar memory segment (like human's associative memory)

        # step 3 : Decide wheter modified and attached to existing memory

        # step 3.1 (case 1) : modified and attached to existing memory

        # step 3.2 (case 2) : Make a new memory
        pass

    def retrieve(self, dialog: List, retrieval_k: int = 3):
        """
        Given current instruction to retrieve relevent document(memory) from the bank(self)
        """
        retrieved_memory_segment = ""
        history = dialog_to_string(dialog=dialog)

        return retrieved_memory_segment


class SelfDocAgent:
    def __init__(self, model: str = "gpt-4-0613"):
        # init model
        self.model = model
        self.llm_interpreter = GPTCodeInterpreter(model=model)
        self.INIT_SYS_PROMPT = self.llm_interpreter.dialog[0]["content"]

        # init memory
        self.memories = MemoryBank(model=model)

    def close(self):
        self.llm_interpreter.close()

    def step(
        self,
        instruction: str,
        MAX_TRY: int = 7,
        temperature: float = 0.1,
        top_p: float = 0.95,
        VERBOSE: bool = True,
    ):
        # step 1 : retrieve document first
        document = self.memories.retrieve(
            dialog=self.llm_interpreter.dialog
            + [{"role": "user", "content": instruction}]
        )

        # Add document on SYS PROMPT
        extra_sys_prompt = (
            f"\nFrom prior experience you documented :\n{document}"
            if document
            else "None"
        )
        self.llm_interpreter.dialog[-1]["content"] = (
            self.INIT_SYS_PROMPT + extra_sys_prompt
        )

        # step 2 : agent make trajectories
        self.llm_interpreter.chat(
            user_message=instruction,
            MAX_TRY=MAX_TRY,
            temperature=temperature,
            top_p=top_p,
            VERBOSE=VERBOSE,
        )

        # step3 : encode it's experience
        self.memories.encode(dialog=self.llm_interpreter.dialog)

        return self.llm_interpreter.dialog[-1]


if __name__ == "__main__":
    agent = SelfDocAgent()

    agent.step(
        instruction="what's the weather in korea right now?",
        VERBOSE=True,
    )

    agent.close()
