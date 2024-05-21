# LLM Prompt Strategies for Commonsense-Reasoning Tasks

## Description

This project aims to investigate and compare various prompt strategies, such as Chain of Thought (CoT), in-context learning, and plan-and-solve techniques, to enhance commonsense reasoning in Large Language Models (LLMs). Using different LLM frameworks and datasets, we evaluate the effectiveness of each strategy. The goal is to understand how different prompting techniques influence LLM performance on tasks reliant on commonsense knowledge.

This project is being developed as part of the *Natural Language Processing* course at the *Faculty of Computer and Information Science* in Ljubljana.

## Repository structure


**/src**

The /src folder will contain the scripts for running the project.
It is currently empty.

**/report**

In the report folder, you will find the file [report.pdf](report/report.pdf), which is our report for the first phase of the project.

## Usage
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-yugogpt.git
```
Then, run the following commands:
```bash
cd ul-fri-nlp-course-project-yugogpt
conda create --name nlp_env --file req.txt
conda activate nlp_env
```

### Different prompting strategies
We have implemented following existing prompting strategies: no prompting, zero-shot chain of thought, plan & solve, chain of thought.
To try those prompting strategies, edit the file lamma.py, by changing *strategy* variable. 

Based on the PromptBreeder, we have created three new prompting strategies: argumentative, using mutations, and using thinking styles.
To try those prompting strategies, edit the file lamma_llm.py, by changing *strategy* variable. 

### Different datasets
We have available two multiple-choice datasets: CommonsenseQA and StrategyQA, and one generative dataset ProtoQA. 
To try different datasets, edit files lamma.py / lamma_llm.py, by changing *dataset_name* variable.

## Contributors
- Jaša Samec (js7039@student.uni-lj.si)
- Haris Kupinić (hk8302@student.uni-lj.si)
- Jovana Videnović (jv8043@student.uni-lj.si)

Supervisor: Assist. Aleš Žagar
