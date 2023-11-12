from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from prompt.chatgpt_explore_prompt import exploration_instruction
import json


def init_tool_prompt():
    tool_init_json_path = "./tools/tool_init.json"
    with open(tool_init_json_path, "r") as file:
        data = json.load(file)

    prompt = ""
    for k, v in data.items():
        prompt += f"**[{k}]**\nurl:{v['url']}\nsub_url:{v['sub_url']}\nAPI_ENV_VAR_NAME:{v['API_KEY_NAME']}"

    return prompt


def init_tool_prompt_by_sub():
    tool_init_json_path = "./tools/tool_init.json"
    with open(tool_init_json_path, "r") as file:
        data = json.load(file)

    endpoints = list()
    for k, v in data.items():
        for sub in v["sub_url"]:
            endpoints.append(
                f"**[{k}]**\n+root url : {v['url']}\n+ endpoint : {sub['endpoint']}\n+ example query : {sub['example_query']}\n+ API_ENV_VAR_NAME:{v['API_KEY_NAME']}"
            )

    return endpoints


if __name__ == "__main__":
    tool_init_prompt = init_tool_prompt_by_sub()[0]

    gpt_interpreter = GPTCodeInterpreter()

    answer = gpt_interpreter.chat(
        exploration_instruction.format(tool_init_prompt=tool_init_prompt),
        MAX_TRY=10,
        temperature=0.2,
        top_p=0.9,
    )

    gpt_interpreter.close()

    # print(answer)
