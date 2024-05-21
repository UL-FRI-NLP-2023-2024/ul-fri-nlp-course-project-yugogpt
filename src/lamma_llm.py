import transformers
from transformers import AutoTokenizer

import json
import torch
from prompt_creator import PromptCreator
from utils_funs import load_data, extract_answer

# define the model
model = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# define dataset
dataset_name = "commonsenseqa" # "strategyqa", "protoqa"

# define strategy
strategy = "argumentative" # "llm_arg", "no_prompting", "plan_and_solve", "zero_shot", "cot", "llm1"

# load data
q, a, ids = load_data(dataset_name)

# create prompt
prompt_creator = PromptCreator(strategy, (q, a, ids))

res = []
for i in range(len(q)):
    print(f"Processing question {i+1}/{len(q)}")
    prompt = prompt_creator.get_next_prompt()
    
    # This is because i don't want to load two models at the same time
    if "llm" == strategy: 
        prompt, anwser = prompt_creator.get_next_prompt()
        #prompt = prompt_creator.get_next_prompt()
        mutator = prompt
        
        print(prompt)
        
        
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        
        response = sequences[0]['generated_text']
        
        print(response)
        
        new_prompt = response.split("New Q:")[1]
        new_prompt = "Q: " + new_prompt + "\n"
        new_prompt = prompt_creator.wrap_in_instructions(new_prompt + anwser)
        new_prompt = new_prompt + "A:"
        prompt = new_prompt
        
    if "llm_arg" == strategy: 
        prompt, question = prompt_creator.get_next_prompt()
        mutator = prompt

        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        
        response = sequences[0]['generated_text']
        
        new_prompt = "Answer the following questions. You may use the arguments provided but only one answer is correct. \n"
        new_prompt = new_prompt + question + "\n"
        
        arguments = response.split("Arguments:")[1]
        new_prompt = new_prompt + "Arguments:" + arguments
        new_prompt = prompt_creator.wrap_in_instructions(new_prompt)
        new_prompt = new_prompt + "A:"        
        prompt = new_prompt
        
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )

    response = sequences[0]['generated_text']
    if strategy == "cot":   
        response_list = response.split("A2:")
        question = response_list[0]
        response = response_list[1]
    else:
        response = response.split("[/INST]\nA:")[1:]
        # join the response
        response = " ".join(response)


    try:

        pred_answer = extract_answer(dataset_name, response)
    except Exception as e:
        print(f"Error: {e}")
        pred_answer = "?"
    
    res_dict = {}
    res_dict["ID"] = ids[i]
    res_dict["question"] = q[i] if strategy != "cot" else question
    res_dict["whole_pred"] = response
    res_dict["pred"] = pred_answer
    res_dict["answer"] = a[i]
    res_dict["ans"] = a[i] == pred_answer
    
    if "llm" in strategy:
        res_dict["mutator"] = mutator
    
    res.append(res_dict)

    # save results to json
    with open(f"../generated_result/{strategy}/{dataset_name}.json", "w") as f:
        json.dump(res, f)

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
