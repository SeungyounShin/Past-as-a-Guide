from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from utils.utils import *

import os, sys, json
import io
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
import copy
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

        # memory organizer
        self.organizer = "gpt-4"

    def __len__(self):
        return len(self.memories)

    def update_memory_by_id(self, memory):
        # just linear search

        for idx, mem in enumerate(self.memories):
            if memory["id"] == mem["id"]:
                original_mem = copy.deepcopy(self.memories[idx])
                self.memories[idx] = {
                    "id": self.memories[idx]["id"],
                    "query": memory["query"],
                    "retrospection": memory["retrospection"],
                    "embedding_file": self.memories[idx]["embedding_file"],
                    "embedding": memory["embedding"],
                }
                self.console.print(
                    f"[bold]Memory Updated \n{original_mem} -> {self.memories[idx]}[/bold]"
                )
                return True

        return False

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(30))
    def memory_organizer(self, prompt: str):
        response = openai.ChatCompletion.create(
            model=self.organizer,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI specialist, streamlining solutions by comparing a current problem's retrospection with archival memories. Given the recent problem retrospection and related past memories, follow these steps:\n\n1. **Assessment**: Contrast the current retrospection with the supplied memories. Pinpoint the singular most similar memory to the ongoing problem.\n2. **Validation**: Ascertain if the present retrospection introduces novel or enhanced information compared to the selected memory.\n3. **Enhancement & Clarification**: If the current retrospection proves more advanced, amalgamate this newfound knowledge into the chosen memory. Subsequently, unambiguously declare if the resulting retrospection will supersede the previous one or be annexed as a distinct new entry. Ensure to encapsulate the final retrospection with:\n```retrospection\n(content)\n```\nMoreover, indicate the placement for the enhanced retrospection (if not being added, simply state `-1`) using:\n```choosen\nm\n```\nif m is not -1 then also adjust the query of m of <3 similar memories>\n```query-rewrite\n(content)\n```",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        response_text = response["choices"][0]["message"]["content"]
        # Extracting retrospection
        retrospection_match_sample = re.search(
            r"```retrospection\s*\n(.*?)\n```", response_text, re.DOTALL
        )
        retrospection_text = (
            retrospection_match_sample.group(1).strip()
            if retrospection_match_sample
            else None
        )

        choosen_match_sample = re.search(
            r"```choosen\s*\n(.*?)\n```", response_text, re.DOTALL
        )
        choosen_text = (
            choosen_match_sample.group(1).strip() if choosen_match_sample else None
        )

        query_match_sample = re.search(
            r"```query-rewrite\s*\n(.*?)\n```", response_text, re.DOTALL
        )
        query_rewrite_text = (
            query_match_sample.group(1).strip() if query_match_sample else None
        )

        self.console.print(
            f"✅ [green]Memory Organized.[green]\n{response_text}\n-----\n"
        )
        return choosen_text, retrospection_text, query_rewrite_text

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
                with open(
                    os.path.join(self.memory_dir_path, embedding_file), "rb"
                ) as f:
                    buffer = f.read()
                    embedding = np.load(io.BytesIO(buffer))
                data["embedding"] = embedding
                self.memories.append(data)

        self.console.print(
            Markdown(f"## ✅ Successfully loaded {len(self.memories)} memory files!")
        )

    def save_memory(self, memory, embedding, memory_id: Optional[int] = None):
        if memory_id is None:
            memory_id = len(self.memories) + 1
        embedding_file = f"memory_{memory_id}_embedding.npy"
        np.save(os.path.join(self.memory_dir_path, embedding_file), embedding)

        memory["embedding_file"] = embedding_file
        json_file = f"memory_{memory_id}.json"
        with open(os.path.join(self.memory_dir_path, json_file), "w") as f:
            json.dump(memory, f)

    def encode(
        self,
        init_retrieved,
        retrospection_dict,
    ):
        if init_retrieved is None:
            self.console.print(f"No memory to encode (no retrieved memory exists)")
            return

        operation = retrospection_dict["operation"]
        if len(init_retrieved) == 0:
            operation = "<new>"  # when no retrieved memory it should be new
        if (retrospection_dict["query"] is None) or (
            len(retrospection_dict["query"]) == 0
        ):
            # query is too short (no content)
            return False
        query_embedding = get_embedding_ada(retrospection_dict["query"])

        # memory encoding by operation
        if operation == "<new>":
            # adding new memory
            memory = {
                "id": len(self.memories) + 1,
                "query": retrospection_dict["query"],
                "retrospection": retrospection_dict["retrospection"],
            }
            self.console.print(f"Memory Encoded : [bold][green]<new>[/green][bold]")
            self.save_memory(memory, query_embedding)
            self.update_memory()
            return True
        elif operation == "<revise>":
            # revise initial retrieved memory
            ## save revised memory (in disk)
            memory = {
                "id": init_retrieved[0]["id"],
                "query": retrospection_dict["query"],
                "retrospection": retrospection_dict["retrospection"],
            }
            self.save_memory(
                memory,
                query_embedding,
                memory_id=init_retrieved[0]["id"],
            )
            ## update working memory (in RAM)
            self.console.print(f"Memory Encoded : [bold][green]<revised>[/green][bold]")
            memory["embedding"] = query_embedding
            self.update_memory_by_id(memory)

        elif operation == "<merge>":
            # merge not retrieved but similar k-th retrieved memory
            merge_id = 1
            memory = {
                "id": init_retrieved[merge_id]["id"],
                "query": retrospection_dict["query"],
                "retrospection": retrospection_dict["retrospection"],
            }
            self.save_memory(
                memory,
                query_embedding,
                memory_id=init_retrieved[merge_id]["id"],
            )
            ## update working memory (in RAM)
            memory["embedding"] = query_embedding
            self.update_memory_by_id(memory)
            self.console.print(f"Memory Encoded : [bold][green]<merge>[/green][bold]")

        elif operation == "<none>":
            # not doing anything
            return False
        else:
            # not permitted memory operation
            self.console.print(f"Memory Encoded : [bold][green]<none>[/green][bold]")
            return False

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
            memory_idx_to_load = list(
                range(len(self.memories) + 1, len(json_files) + 1)
            )
            for i in tqdm(memory_idx_to_load):
                with open(
                    os.path.join(self.memory_dir_path, f"memory_{i}.json"), "r"
                ) as f:
                    data = json.load(f)
                    embedding_file = data["embedding_file"]
                    with open(
                        os.path.join(self.memory_dir_path, embedding_file), "rb"
                    ) as f:
                        buffer = f.read()
                        embedding = np.load(io.BytesIO(buffer))
                    data["embedding"] = embedding
                    self.memories.append(data)

            self.console.print(f"✅ [green]Updated newly encoded memory...[green]")
            self.console.print(f"[blue]Total Memory : {len(self.memories)}[blue]")


class SelfDocAgent:
    def __init__(self, model: str = "gpt-4-0613", memory: Optional[MemoryBank] = None):
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

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
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

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def get_retrospection_v2(
        self,
        dialog: List[Dict],
        init_retrieved: Optional[List] = None,
        VERBOSE: bool = True,
        ORGANIZE_MEMORY: bool = True,
    ):
        full_traj = dialog_to_string(dialog, pretty=True)

        if (init_retrieved is None) or len(init_retrieved) <= 0:
            full_prompt = (
                f"<From prior experience>:\n\nNone\n\n"
                f"<Agent Log>:\n\n{full_traj}\n\n"
                f"<Extra memories not chosen>:\n\nNone\n"
            )

        elif (init_retrieved is not None) and len(init_retrieved) >= 1:
            retrospect = init_retrieved[0]
            extra_mem = init_retrieved[1:]
            full_prompt = (
                f"<From prior experience>:\n\nTitle :\n{retrospect['query']}\n\nContent :\n{retrospect['retrospection']}\n\n"
                f"<Agent Log>:\n\n{full_traj}\n\n"
                f"<Extra memories not chosen>:\n\n"
                + "\n".join(
                    f"[{index}]\n\nTitle: \n{memory['query']}\n\nContent :\n{memory['retrospection']}"
                    for index, memory in enumerate(extra_mem)
                )
            )

        else:
            raise f"Error during retrospection\ninit_retrieved : {init_retrieved}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": 'Given the provided AGENT_LOG, prior experiences, and extra memories, you are tasked with deciding the best memory encoding operation to use. \n\nYour output should consist of the following:\n\n1. Reflect upon the interaction in the <agent log> and evaluate if there is any newfound knowledge or insights that were not previously captured in <From prior experience>. Additionally, assess whether any information from <Extra Memories not considered> should be integrated or utilized for a more comprehensive understanding.\n\n\n2. **Memory Encoding Operation Choice**:\n   - `<new>`: If the information from "From prior experience" is not deemed helpful and a new retrospection is needed.\n   - `<revise>`: If there\'s a discovered advanced way of solving the tasks, so revise the query and retrospection of "From prior experience".\n   - `<merge>`: If neither "From prior experience" nor a specific memory from "Extra memories not chosen" is helpful, then merge with an unused memory.\n   - `<none>`: If the previous retrospections are helpful and no further improvement is required.\n\n3. **Action Based on Operation**:\n\n   - For `<new>`, provide a new query and retrospection.\n   - For `<revise>`, give the revised query and retrospection.\n   - For `<merge>`, specify which memory (m-th memory) from "Extra memories not chosen" is used and the merged retrospection.\n   - If `<none>`, specify that no further action is needed.\n\nMake sure to encapsulate \n```query\ncontent\n```\n```retrospection\ncontent\n```',
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["message"]["content"]

        op_choosen, query, retrospection = self.extract_retrospect(response)

        if VERBOSE:
            self.console.print(Markdown(f"## Full Retrospection : \n{response}\n\n"))

        return {
            "operation": op_choosen if op_choosen else "<none>",
            "query": query if query else "",
            "retrospection": retrospection if retrospection else "",
        }

    def extract_query_and_key(response: str):
        # Extract retrospection content including nested content
        retrospection_pattern = r"```retrospection((?:[^`]+|`+(?!``))*)```"
        retrospection_match = re.search(retrospection_pattern, response, re.DOTALL)
        retrospection_content = (
            retrospection_match.group(1).strip() if retrospection_match else None
        )

        # Extract the user's core problem content
        core_problem_pattern = r"1\.\s(.*?)(\.|\n|$)"
        core_problem_match = re.search(core_problem_pattern, response)
        core_problem_content = (
            core_problem_match.group(1).strip() if core_problem_match else None
        )

        return core_problem_content, retrospection_content

    def extract_retrospect(self, text: str):
        ops = ["<merge>", "<new>", "<none>", "<revise>"]
        op_choosen = "<none>"
        op_num = 999999
        for op in ops:
            _op_num = text.find(op)
            if (_op_num >= 0) and (_op_num < op_num):
                op_num = _op_num
                op_choosen = op

        query_match = re.search(r"query\r?\n(.*?)\r?\n", text, re.DOTALL)
        retrospection_match = re.search(
            r"retrospection\r?\n(.*?)\r?\n", text, re.DOTALL
        )

        query = query_match.group(1).strip() if query_match else None
        retrospection = (
            retrospection_match.group(1).strip() if retrospection_match else None
        )

        return (op_choosen, query, retrospection)

    def step(
        self,
        instruction: str,
        MAX_TRY: int = 7,
        temperature: float = 0.1,
        top_p: float = 0.95,
        VERBOSE: bool = True,
        USE_RETRIEVE: bool = True,
        USE_ENCODE: bool = True,
        ORGANIZE_MEMORY: bool = True,
        MULTITURN: bool = False,
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
        if MULTITURN:
            second_round = input(" User >")
            self.llm_interpreter.chat(
                user_message=second_round,
                MAX_TRY=MAX_TRY,
                temperature=temperature,
                top_p=top_p,
                VERBOSE=VERBOSE,
            )

        # step3 : encode it's experience
        if USE_ENCODE:
            retrospection_dict = self.get_retrospection_v2(
                dialog=self.llm_interpreter.dialog,
                init_retrieved=retrieved,
                VERBOSE=VERBOSE,
                ORGANIZE_MEMORY=ORGANIZE_MEMORY,
            )
            self.memories.encode(
                init_retrieved=retrieved,
                retrospection_dict=retrospection_dict,
            )

        return self.llm_interpreter.dialog


if __name__ == "__main__":
    agent = SelfDocAgent()

    problem_example = """
Can you draw cute cat for me? with stable diffusion?

"""

    agent.step(
        instruction=problem_example,
        USE_RETRIEVE=True,
        USE_ENCODE=True,
        ORGANIZE_MEMORY=True,
        VERBOSE=True,
        MULTITURN=True,
    )

    agent.save_traj("stablediffusion_1")
    agent.close()
