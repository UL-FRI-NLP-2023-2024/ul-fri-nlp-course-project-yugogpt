import json
import os
import random

from utils_funs import load_data


class PromptCreator:

    INSTRUCTION_START = "[INST]"
    INSTRUCTION_END = "[/INST]"

    def __init__(self, strategy, data):
        self.strategy = strategy
        self.current_index = 0
        self.q, self.a, self.ids = data
        self.length = len(self.q)
        if strategy == "cot":
            self.q_cot, self.a_cot, self.ids_cot, self.wa_cot = load_data("commonsenseqa_cot")

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

    def CoT(self, question):
        # get a random COT example
        while True:
            cot_idx = random.randint(0, len(self.q_cot) - 1)
            if self.q_cot[cot_idx] != question:
                break

        cot_prompt = ""
        cot_intro = "You are a question and answer master. In the input you have two questions. One has a correct response and the other has no response. Your job is to answer the question that has no response."
        cot_q1 = cot_intro + "\nQ1: " + self.q_cot[cot_idx] + "\n"
        cot_q1 = self.wrap_in_instructions(cot_q1)

        whole_answer = self.wa_cot[cot_idx].replace("A: ", "")
        cot_a1 = "\nA1: " + whole_answer

        cot_q2 = "Q2: " + question + "\n"
        cot_q2 = "\n" + self.wrap_in_instructions(cot_q2)

        cot_a2 = "\nA2:"

        cot_prompt = cot_q1 + cot_a1 + cot_q2 + cot_a2
        return cot_prompt

    def wrap_in_instructions(self, text):
        return self.INSTRUCTION_START + "\n" + text + self.INSTRUCTION_END + "\n"

    def create_prompt(self, question, extra_instructions=False):
        if self.strategy == "cot":
            prompt = self.CoT(question)
        else:
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
    strategy = "cot"  # "plan_and_solve", "zero_shot"

    # load data
    q, a, ids = load_data(dataset_name)

    # create prompt
    prompt_creator = PromptCreator(strategy, (q, a, ids))

    res = []
    for i in range(1):  # range(len(q)):
        print(f"Processing question {i+1}/{len(q)}")
        prompt = prompt_creator.get_next_prompt(add_beginning_of_answer=True)
        print(prompt)
