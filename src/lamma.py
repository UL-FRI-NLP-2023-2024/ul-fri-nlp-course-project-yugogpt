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
dataset_name = "strategyqa"

# load data
q, a, ids = load_data(dataset_name)

# create prompt
prompt_creator = PromptCreator("zero_shot", (q, a, ids))

res = []
for i in range(len(q)):
    print(f"Processing question {i+1}/{len(q)}")
    prompt = prompt_creator.get_next_prompt(add_beginning_of_answer=True)

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
    try:
        pred_answer = extract_answer(dataset_name, response)
    except:
        pred_answer = "?"
    
    # print(response)
    res_dict = {}
    res_dict["ID"] = ids[i]
    res_dict["question"] = q[i]
    res_dict["pred"] = pred_answer
    res_dict["answer"] = a[i]
    res_dict["ans"] = a[i] == pred_answer
    res.append(res_dict)

    # save results to json
    with open(f"../generated_result/zero_shot/{dataset_name}.json", "w") as f:
        json.dump(res, f)

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")