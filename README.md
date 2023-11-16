### Past as a Guide


- [NeurIPS 2023 Workshop on Instruction Following](https://neurips.cc/virtual/2023/workshop/66498)
- **Research Paper:** [arXiv](https://arxiv.org/abs/2311.07635)
- **Project Page:** To be uploaded

#### Overview
This project introduces a novel approach to enhancing language model performance by enabling past experience and memory management.

#### Usage

0. Installing the requirments
```bash
python -m pip install -r requirments
```

1. Setting the OPENAI_API
```bash
echo "OPENAI_API_KEY=<YOUR_API_KEY>" >> .env
```

2. Run the agent
```bash
python -m agent.self_doc_agent
```

#### Experiments

- **Legacy Code:** Original implementation available at [GitHub](https://github.com/SeungyounShin/Llama2-Code-Interpreter).
- **New Feature:** Memory operations (add, revise, none) have been incorporated to allow the model to update its experiences and organize its memory effectively.

1. **HumanEval Benchmark:**
   
   | Method | % Pass@1 |
   | ------ | -------- |
   | GPT-4 | 67.00 |
   | GPT-4 + † | 90.85 |
   | Reflexion [17] | 91.00 |
   | GPT-4 + † + PaG | 92.68 |

   † denotes the addition of a code interpreter capability.

2. **DS-1000 Benchmark (Not in Paper):**

   | Condition | pd(origin) | pd(difficult) | All Memory |
   | --------- | ---------- | ------------- | ---------- |
   | No Memory | 0.45 | 0.32 | - |
   | Memory Add Only | 0.46 | 0.34 | 179/179 |
   | PaG (Origin First) | 0.5 | 0.42 | 111/179 |
   | PaG (Difficult First)  | 0.34 | 0.46 | 114/179 |

   Note: The self-updating memory emphasizes learning from easier to more difficult problems to update insightful memory.

#### Key Takeaway
...