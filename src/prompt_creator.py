import json
import os
import random

from utils_funs import load_data


class PromptCreator:

    INSTRUCTION_START = "[INST]"
    INSTRUCTION_END = "[/INST]"

    def __init__(self, strategy, data, dataset_name="commonsenseqa"):
        self.strategy = strategy
        self.current_index = 0
        self.q, self.a, self.ids = data
        self.length = len(self.q)
        
        if "llm" in strategy:
            self.mutations = json.load(open("../datasets/mutator_prompts.json"))[0]
            self.thinking_styles = json.load(open("../datasets/thinking_styles.json"))[0]
        
        if strategy == "cot":
            if dataset_name == "commonsenseqa":
                self.q_cot, _, _, self.wa_cot = load_data("commonsenseqa_cot")
            else:
                self.q_cot, _, _, self.wa_cot = load_data("proto_cot")

    def get_next_prompt(self, extra_instructions=None):
        question = self.q[self.current_index]
        self.current_index += 1
        return self.create_prompt(question, extra_instructions)

    def zero_shot_CoT(self):
        return "A: Let's think step by step."

    def argumentative_CoT(self):
        return "A: Let's first consider arguments for and against each choice and then decide which one is the most convincing."

    def plan_and_solve(self):
        return "A: Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."

    def no_prompting(self):
        return "A:"
    
    def CoT(self, question):
        # get a random COT example
        while True:
            cot_idx = random.randint(0, len(self.q_cot) - 1)
            if self.q_cot[cot_idx] != question:
                break

        cot_prompt = ""
        cot_intro = "" #"You are a question and answer master. In the input you have two questions. One has a correct response and the other has no response. Your job is to answer the question that has no response."
        cot_q1 = cot_intro + "\nQ1: " + self.q_cot[cot_idx] + "\n"
        cot_q1 = self.wrap_in_instructions(cot_q1)

        whole_answer = self.wa_cot[cot_idx].replace("A: ", "")
        cot_a1 = "\nA1: " + whole_answer

        cot_q2 = "Q2: " + question + "\n"
        cot_q2 = "\n" + self.wrap_in_instructions(cot_q2)

        cot_a2 = "\nA2:"

        cot_prompt = cot_q1 + cot_a1 + cot_q2 + cot_a2
        return cot_prompt

    def get_mutation(self):
        key = random.choice(list(self.mutations.keys()))
        return self.mutations[key]
        
    def LLM1(self, question):        
        question, anwsers = question.split("Answer Choices:")

        if len(anwsers) > 0:
            preface = "Help a system solve the multiple-choice questions by modifying/adding instructions based on the following rule:"
        else:
            preface = "Help a system solve the question by modifying/adding instructions based on the following rule:"

        
        mutation = self.get_mutation()
        
        pre_prompt = f"{preface}\n{mutation}\n\nQ: {question}\n"
        pre_prompt = self.wrap_in_instructions(pre_prompt)
        pre_prompt = pre_prompt + "New Q:"
        
        return pre_prompt, "Possible answers: " + anwsers if len(anwsers) > 0 else ""

    
    def get_thinking_style(self):
        key = random.choice(list(self.thinking_styles.keys()))
        return self.thinking_styles[key]
    
    def LLM_T(self, question):
        question, anwsers = question.split("Answer Choices:")
        
        preface = "You are a teacher and want to help students solve the multiple-choice questions by adding a thinking style. For the question below, you need to provide a tip to help students solve it."
    
        thinking_style = self.get_thinking_style()
        
        pre_prompt = f"{preface}\nThinking Style:{thinking_style}\nQuestion to solve:\nQ: {question}\n"
        pre_prompt = self.wrap_in_instructions(pre_prompt)
        pre_prompt = pre_prompt + "New Q:"
        
        return pre_prompt, "Possible answers: " + anwsers if len(anwsers) > 0 else ""

    def LLMARG(self, question):        
        question, anwsers = question.split("Answer Choices:")
        
        prompt_start = "You are a teacher in a debate club. For the following multiple-choice question, you need to provide a short and compelling argument for each of the answer choices. The question is:"
        prompt = f"{prompt_start}\n\nQ: {question}\n"
        
        if len(anwsers) > 0:
            prompt_middle = "Now, provide a compelling argument for each of the answer choices:"
            prompt = f"{prompt}\n{prompt_middle}\n{anwsers}\n"
        
        prompt = self.wrap_in_instructions(prompt)
        
        prompt = prompt + "Arguments:"
                
        return prompt, "Q: " + question + "\n" + "A: " + anwsers  if len(anwsers) > 0 else ""

    def wrap_in_instructions(self, text):
        return self.INSTRUCTION_START + "\n" + text + self.INSTRUCTION_END + "\n"

    def create_prompt(self, question, extra_instructions=False):
        if self.strategy == "cot":
            prompt = self.CoT(question)
        elif self.strategy == "llm_mutate":    
            prompt = self.LLM1(question)
        elif self.strategy == "llm_arg":
            prompt = self.LLMARG(question)
        elif self.strategy == "llm_thinking":
            prompt = self.LLM_T(question)    
        else:
            stem_and_choices = self.wrap_in_instructions(question)
            prompt = ""
            if self.strategy == "zero_shot":
                start_of_answer = self.zero_shot_CoT()
                prompt = stem_and_choices + start_of_answer
            elif self.strategy == "argumentative":
                start_of_answer = self.argumentative_CoT()
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
    dataset_name = "protoqa" 
    strategy = "cot"

    q, a, ids = load_data(dataset_name)

    prompt_creator = PromptCreator(strategy, (q, a, ids), dataset_name)

    for i in range(1):
        prompt = prompt_creator.get_next_prompt()
        print(prompt)
