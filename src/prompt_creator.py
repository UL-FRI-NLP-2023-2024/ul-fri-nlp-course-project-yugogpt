import json
import os
import random

from utils_funs import calculate_accuracy, load_data


class PromptCreator:

    INSTRUCTION_START = "[INST]"
    INSTRUCTION_END = "[/INST]"

    def __init__(self, strategy, data):
        self.strategy = strategy
        self.current_index = 0
        self.q, self.a, self.ids = data
        self.length = len(self.q)

    def get_next_prompt(self, extra_instructions=None, add_beginning_of_answer=False):
        question = self.q[self.current_index]
        self.current_index += 1
        return self.create_prompt(question, extra_instructions)

    def zero_shot_CoT(self):
        return "A: Let's think step by step."

    def plan_and_solve(self):
        return "A: Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."

    def no_prompting():
        return "A:"

    def wrap_in_instructions(self, text):
        return self.INSTRUCTION_START + "\n" + text + self.INSTRUCTION_END + "\n"

    def create_prompt(self, question, extra_instructions):
        stem_and_choices = self.wrap_in_instructions(question)

        prompt = ""

        if self.strategy == "zero_shot":
            start_of_answer = self.zero_shot_CoT()
            prompt = stem_and_choices + start_of_answer
        elif self.strategy == "plan_and_solve":
            start_of_answer = self.plan_and_solve()
            prompt = stem_and_choices + start_of_answer
        elif self.strategy == "no_prompting":
            start_of_answer = self.no_prompting()
            prompt = stem_and_choices + start_of_answer

        if extra_instructions:
            prompt = self.wrap_in_instructions(extra_instructions) + prompt

        return prompt


if __name__ == "__main__":
    # define dataset
    dataset_name = "commonsenseqa"  # "strategyqa"

    # define strategy
    strategy = "zero_shot"  # "plan_and_solve", "zero_shot"

    # load data
    q, a, ids = load_data(dataset_name)

    # create prompt
    prompt_creator = PromptCreator(strategy, (q, a, ids))

    evaluate = True

    # res = []
    # for i in range(3):  # range(len(q)):
    #     print(f"Processing question {i+1}/{len(q)}")
    #     prompt = prompt_creator.get_next_prompt(add_beginning_of_answer=True)
    #     print(prompt)

    if evaluate:
        calculate_accuracy()
