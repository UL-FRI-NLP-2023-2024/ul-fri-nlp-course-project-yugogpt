import transformers
from transformers import AutoTokenizer

import json
import torch
from prompt_creator import PromptCreator
from utils_funs import load_data, extract_answer

# define the model
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# define dataset
dataset_name = "protoqa" # "strategyqa"

# define strategy
strategy = "cot" # "no_prompting", "plan_and_solve", "zero_shot" "cot"

# load data
q, a, ids = load_data(dataset_name)

# create prompt
prompt_creator = PromptCreator(strategy, (q, a, ids))

res = []
for i in range(len(q)):
    print(f"Processing question {i+1}/{len(q)}")
    prompt = prompt_creator.get_next_prompt()

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
        if dataset_name == "protoqa":
            pred_answer = "X"
        else:
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
    res.append(res_dict)

    # save results to json
    with open(f"../generated_result/{strategy}/{dataset_name}.json", "w") as f:
        json.dump(res, f)

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

