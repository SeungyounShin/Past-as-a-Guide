from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from utils.utils import *

import os, sys, json
import io
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from retrying import retry
import openai

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress


class MemoryBank:
    def __init__(self, MEMORY_DIR_PATH: str):
        # for printing
        self.console = Console()
        self.console.print(Markdown(f"## 🚀 Initializing MemoryBank..."))

        script_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(script_directory)

        self.memory_dir_path = MEMORY_DIR_PATH

        if not os.path.exists(self.memory_dir_path):
            os.makedirs(self.memory_dir_path)

        # Load memories from dir
        self.memories = []
        self.load_memories()

    def __len__(self):
        return len(self.memories)

    def load_memories(self):
        json_files = [
            f
            for f in os.listdir(self.memory_dir_path)
            if f.endswith(".json") and ("memory_" in f)
        ]

        self.console.print(Markdown(f"## 📁 Loading memories from directory..."))

        for json_file in tqdm(json_files):
            with open(os.path.join(self.memory_dir_path, json_file), "r") as f:
                data = json.load(f)
                embedding_file = data["embedding_file"]
                with open(os.path.join(self.memory_dir_path, embedding_file), "rb") as f:
                    buffer = f.read()
                    embedding = np.load(io.BytesIO(buffer))
                data["embedding"] = embedding
                self.memories.append(data)

        self.console.print(
            Markdown(f"## ✅ Successfully loaded {len(self.memories)} memory files!")
        )

    def save_memory(self, memory, embedding):
        memory_id = len(self.memories) + 1
        embedding_file = f"memory_{memory_id}_embedding.npy"
        np.save(os.path.join(self.memory_dir_path, embedding_file), embedding)

        memory["embedding_file"] = embedding_file
        json_file = f"memory_{memory_id}.json"
        with open(os.path.join(self.memory_dir_path, json_file), "w") as f:
            json.dump(memory, f)

    def encode(self, query: str, retrospection: str, retrieval_k: int = 3):
        if query is None:
            return
        embedding = get_embedding_ada(query)

        memory = {
            "id": len(self.memories) + 1,
            "query": query,
            "retrospection": retrospection,
        }
        self.save_memory(memory, embedding)

    def retrieve(self, query: str, retrieval_k: int = 3):
        # Get Top k similar memory
        embedding = get_embedding_ada(query)

        # If no memories yet
        if len(self.memories) == 0:
            self.console.print("No memories stored yet!")
            return []

        all_embeddings = np.vstack([memory["embedding"] for memory in self.memories])
        all_embeddings_normed = (
            all_embeddings / np.linalg.norm(all_embeddings, axis=1)[:, np.newaxis]
        )
        query_embedding_normed = embedding / np.linalg.norm(embedding)
        dot_products = np.dot(all_embeddings_normed, query_embedding_normed)
        top_k_indices = dot_products.argsort()[-retrieval_k:][::-1]

        # Retrieve the top-k memories
        retrieved_memories = [self.memories[i] for i in top_k_indices]

        return retrieved_memories

    def update_memory(self):
        
        json_files = [
            f
            for f in os.listdir(self.memory_dir_path)
            if f.endswith(".json") and ("memory_" in f)
        ]
        
        if len(self.memories) != len(json_files):
            memory_idx_to_load = list(range(len(self.memories)+1, len(json_files)+1))
            for i in tqdm(memory_idx_to_load):
                with open(os.path.join(self.memory_dir_path, f"memory_{i}.json"), "r") as f:
                    data = json.load(f)
                    embedding_file = data["embedding_file"]
                    with open(os.path.join(self.memory_dir_path, embedding_file), "rb") as f:
                        buffer = f.read()
                        embedding = np.load(io.BytesIO(buffer))
                    data["embedding"] = embedding
                    self.memories.append(data)
                
            self.console.print(f"✅ [green]Update newly encoded image...[green]")        
            self.console.print(f"[blue]Total Memory : {len(self.memories)}[blue]")        
                

class SelfDocAgent:
    def __init__(self, model: str = "gpt-4-0613", memory:Optional[MemoryBank]=None):
         # print
        self.console = Console()
        
        # init model
        self.model = model
        self.llm_interpreter = GPTCodeInterpreter(model=model)
        self.INIT_SYS_PROMPT = self.llm_interpreter.dialog[0]["content"]

        # init memory
        self.MEMORY_DIR_PATH = os.path.join(
            "./memory", self.__class__.__name__, self.model
        )
        if memory is None:
            self.load_memory()
            self.console.print(Markdown(f"## New Memory Loaded\n"))
        else:
            self.memories = memory
        
        print(f"DEBUG : {len(self.memories)}")

    def load_memory(self):
        self.memories = MemoryBank(MEMORY_DIR_PATH=self.MEMORY_DIR_PATH)
        return self.memories

    def get_memory(self):
        return self.memories

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm_interpreter.__exit__(exc_type, exc_value, traceback)

    def close(self):
        if not hasattr(self, "_is_closed") or not self._is_closed:
            self.llm_interpreter.close()
            self._is_closed = True

    def save_traj(self, traj_name: str):
        # check MEMORY_DIR_PATH exist
        if not os.path.exists(self.MEMORY_DIR_PATH):
            os.makedirs(self.MEMORY_DIR_PATH)

        save_dict = {
            "dialog": self.llm_interpreter.dialog,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": self.llm_interpreter.dialog[-1]["content"],
            "init_system_prompt": self.INIT_SYS_PROMPT,
            # "document": self.document,
        }

        MEMORY_FILE_PATH = os.path.join(
            self.MEMORY_DIR_PATH,
            hashlib.sha1(self.instruction.encode()).hexdigest() + ".json"
            if traj_name == ""
            else traj_name,
        )

        # Save
        with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)

    @retry(
        stop_max_attempt_number=7,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def get_retrospection(self, dialog: List[Dict], VERBOSE: bool = True):
        full_traj = dialog_to_string(dialog, pretty=True)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are retrospectionGPT. \n\nAfter each USER and ASSISTANT interaction, perform the following:\n\n1. Clearly describe the core problem that the user presented.\n2. Evaluate the solution you provided. Was it effective? Why or why not?\n3. If the problem required multiple attempts, identify the attempt where the correct solution was provided.\n\nFinally, summarize your insights as follows:\n```retrospection\n(e.g., For downloading a video from YouTube, utilize `yt-dlp`. Ensure to indicate the desired format in `yt-dlp` if the user has specified one.)\n```\n",
                },
                {"role": "user", "content": f"{full_traj}"},
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["message"]["content"]

        query, retrospection = self.extract_query_and_key(response)

        if VERBOSE:
            self.console.print(Markdown(f"## Full Retrospection : \n{response}\n\n"))
            self.console.print(
                Markdown(
                    f"## Retrospection\n\nQuery:\n{query}\n\nRetrospection:\n{retrospection}"
                )
            )
        return {
            "query": query if query else "",
            "retrospection": retrospection if retrospection else "",
        }

    def extract_query_and_key(self, response: str):
        # Extract retrospection content
        retrospection_pattern = r"```retrospection\s(.*?)```"
        retrospection_match = re.search(retrospection_pattern, response, re.DOTALL)
        retrospection_content = (
            retrospection_match.group(1).strip() if retrospection_match else None
        )

        # Extract the user's core problem content
        core_problem_pattern = r"1\.\s(.*?\.)"
        core_problem_match = re.search(core_problem_pattern, response)
        core_problem_content = (
            core_problem_match.group(1).strip() if core_problem_match else None
        )

        return core_problem_content, retrospection_content

    def step(
        self,
        instruction: str,
        MAX_TRY: int = 7,
        temperature: float = 0.1,
        top_p: float = 0.95,
        VERBOSE: bool = True,
        USE_RETRIEVE: bool = True,
        USE_ENCODE: bool = True,
    ) -> List[Dict]:
        self.instruction = instruction

        # step 1 : retrieve document first
        extra_sys_prompt = ""
        if USE_RETRIEVE:
            retrieved = self.memories.retrieve(self.instruction)
            if len(retrieved) >= 1:
                document = retrieved[0]["retrospection"]
                extra_sys_prompt = (
                    f"\nFrom prior experience you documented :\n{document}"
                )
                if VERBOSE:
                    self.console.print(Markdown(f"## Document Retrieved\n\n{document}"))
            else:
                extra_sys_prompt = ""

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
        if USE_ENCODE:
            retrospection_dict = self.get_retrospection(
                dialog=self.llm_interpreter.dialog, VERBOSE=VERBOSE
            )
            self.memories.encode(
                query=retrospection_dict["query"],
                retrospection=retrospection_dict["retrospection"],
            )

        return self.llm_interpreter.dialog


if __name__ == "__main__":
    agent = SelfDocAgent()

    problem_example = """
Problem:
I have a dataset :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
3     B.com   No
4     B.com   No
5     C.com   No


I want to remove duplicates, i.e. keep first occurence of "url" field, BUT  keep duplicates if the field "keep_if_dup" is YES.
Expected output :
id    url     keep_if_dup
1     A.com   Yes
2     A.com   Yes
3     B.com   No
5     C.com   No


What I tried :
Dataframe=Dataframe.drop_duplicates(subset='url', keep='first')


which of course does not take into account "keep_if_dup" field. Output is :
id    url     keep_if_dup
1     A.com   Yes
3     B.com   No
5     C.com   No


A:
```python
import pandas as pd


df = pd.DataFrame({'url': ['A.com', 'A.com', 'A.com', 'B.com', 'B.com', 'C.com', 'B.com'],
                   'keep_if_dup': ['Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes']})
<Fill Solution Code>
print(result)
```
"""

    agent.step(
        instruction=problem_example,
        USE_RETRIEVE=True,
        USE_ENCODE=True,
        VERBOSE=True,
    )

    agent.close()
