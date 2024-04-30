import os
import json
import random

class PromptCreator:
    
    INSTRUCTION_START = "[INST]"
    INSTRUCTION_END = "[/INST]"
    
    def __init__(self, strategy, file_path):
        self.strategy = strategy
        self.data = self.read_data(file_path)
        self.current_index = 0
        self.length = len(self.data)
        
        self.COT_examples = self.read_CoT_examples()
        
    def read_CoT_examples(self, cot_file_path = "../datasets/CoT.json"):
        with open(cot_file_path) as f:
            data = json.load(f)
        return data
        
    def read_data(self, file_path):
        with open(file_path) as f:
            data = f.readlines()
        return data    
    
    def get_next_prompt(self, extra_instructions = None, add_beginning_of_answer = False):
        question = json.loads(self.data[self.current_index])
        self.current_index += 1
        return self.create_prompt(question, extra_instructions, add_beginning_of_answer)
        
    def stringify_choices(self, choices):
        choice_str = ""
        for choice in choices:
            choice_str += "(" + choice['label'] + ") " + choice['text'] + "\n"
        return choice_str

    def join_stem_and_choices(self, stem, choices):
        return "Q: " + stem + "\n" + self.stringify_choices(choices)
    
    def zero_shot_CoT(self):
        return "A: Let's think step by step." # TODO: add some alternatives? "let's break this down..."
        
    def wrap_in_instructions(self, text):
        return self.INSTRUCTION_START + "\n" + text + self.INSTRUCTION_END + "\n"
    
    def CoT(self):
        #TODO: Add some check to the example is different from the current question
        random_example = random.choice(self.COT_examples)
        
        stem_and_choices = self.join_stem_and_choices(random_example['question']['stem'], random_example['question']['choices'])
                
        pre_prompt = self.wrap_in_instructions(stem_and_choices)
        pre_prompt = pre_prompt + random_example['answer'] + "\n"
                        
        return pre_prompt
    
    def create_prompt(self, question, extra_instructions, add_beginning_of_answer):
        stem = question['question']['stem']
        choices = question['question']['choices']
        stem_and_choices = self.join_stem_and_choices(stem, choices)
        stem_and_choices = self.wrap_in_instructions(stem_and_choices)
        
        prompt = ""
            
        if self.strategy == "zero_shot":
            start_of_answer = self.zero_shot_CoT()
            prompt = stem_and_choices + start_of_answer            
        elif self.strategy == "none":
            prompt = stem_and_choices
            if add_beginning_of_answer:
                prompt = prompt + "A: "
               
        elif self.strategy == "cot":
            example = self.CoT()
            prompt = example + stem_and_choices
            
            if add_beginning_of_answer:
                prompt = prompt + "A: "
                
        elif self.strategy == "llm":
            #USE A LLM TO GENERATE/MUTATE THE PROMPT
            raise NotImplementedError("LLM strategy not implemented yet")
        else:
            raise ValueError("Invalid strategy")
        
        if extra_instructions:
            prompt = self.wrap_in_instructions(extra_instructions) + prompt

        return prompt

if __name__ == "__main__":
    # Idea: Itterate over the questions in the dataset and generate a prompt for each one
    prompt_creator = PromptCreator("none", '../datasets/CommonsenseQA/CommonsenseQA.jsonl')
    print(prompt_creator.get_next_prompt(add_beginning_of_answer=True))

