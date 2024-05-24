import re
import json
import numpy as np

def extract_answer(dataset, text):
    if dataset == "commonsenseqa":
        pred = text.strip()
        pred = pred.replace(". A", "")
        pred = pred.replace(": A", "")
        pred = pred.replace("! A", "")
        
        pred = re.sub(r"\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        
        pred_answer = [i for i in pred if i in ('A|B|C|D|E')]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[-1]
        else:
            pred_answer = "X"
        return pred_answer
    elif dataset == "strategyqa":
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    
def load_data(dataset):
    decoder = json.JSONDecoder()
    questions = []
    answers = []
    whole_answers = []
    ids = []
    if dataset == "commonsenseqa":
        datapath = "../datasets/CommonsenseQA/CommonsenseQA.jsonl"
    elif dataset == "strategyqa":
        datapath = "../datasets/StrategyQA/StrategyQA.json"
    elif dataset == "commonsenseqa_cot":
        datapath = "../datasets/CommonsenseQA/CoT.json"
    elif dataset == "protoqa_":
        datapath = "../datasets/ProtoQA/first_10.txt"

    # read dataset file
    if dataset.lower() in ['strategyqa']:
        with open(datapath) as f:
            if dataset.lower() in ['strategyqa']:
                json_data = json.load(f)["examples"]
            else:
                json_data = json.load(f)

            for idx, line in enumerate(json_data):
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                id = 'temp_{}'.format(idx)
                questions.append(q)
                answers.append(a)
                ids.append(id)
    elif dataset.lower() in ['commonsenseqa', 'commonsenseqa_cot']:
        with open(datapath) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                q = json_res["question"]["stem"].strip() + " " + choice
                a = json_res["answerKey"]

                id = 'temp_{}'.format(idx)
                questions.append(q)
                answers.append(a)
                ids.append(id)

                if dataset == "commonsenseqa_cot":
                    whole_a = json_res["answer"]
                    whole_answers.append(whole_a)
    elif dataset.lower() in ['protoqa_']:
        with open(datapath) as f:
            questions = f.readlines()
            answers = np.zeros(len(questions))
            ids = range(len(questions)) 

    if dataset == "commonsenseqa_cot":
        return questions, answers, ids, whole_answers
    return questions, answers, ids

def construct_input(prompt, text):
    inputs = 'Q:' + text + "\nA: " + prompt
    return inputs

def fix_previous_results(json_res_path, new_json_res_path):
    with open(json_res_path) as f:
        json_res = json.load(f)

    new_json_res = json_res.copy()
    correct_answers = 0
    for res in json_res:
        ans = extract_answer("commonsenseqa", res["whole_pred"])
        if ans == res["answer"]:
            correct_answers += 1
        res["pred"] = ans
        res["ans"] = res["answer"] == res["pred"]

    with open(new_json_res_path, 'w') as f:
        json.dump(new_json_res, f, indent=4)
    
    print("Correct answers: ", correct_answers)
    print("Total answers: ", len(json_res))
    print("Solve rate: ", correct_answers / len(json_res))

if __name__ == "__main__":
    # text = "Therefore, the most convincing answer is (B) chess game. A pawn is a versatile piece in chess because of its unique movements and abilities, which make it an important part of the game."
    # ans = extract_answer("commonsenseqa", text)
    # print(ans)

    fix_previous_results("../generated_result/self_argument/self_argument.json", "../generated_result/self_argument/self_argument_new.json")
