import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "w"
plt.rcParams["grid.linestyle"] = "dashed"
plt.rcParams["grid.linewidth"] = 0.5

ROOT_DIR = str(Path(__file__).resolve().parents[1])


def calculate_accuracy():
    dataset = "commonsenseqa.json"
    strategies = ["no_prompting", "zero_shot", "cot", "plan_and_solve"]
    results_path = ROOT_DIR + "/generated_result"

    _map_names = {
        "no_prompting": "Raw Prompt",
        "zero_shot": "Zero-Shot CoT",
        "cot": "Chain-of-Thought",
        "plan_and_solve": "Plan-and-Solve",
    }
    results = {}
    for strategy in strategies:
        with open(results_path + f"/{strategy}/{dataset}", "r") as f:
            result = json.load(f)

            acc = np.mean([result[i]["ans"] for i in range(len(result))])

        results[_map_names[strategy]] = acc

    fig, ax = plt.subplots()

    bars = ax.bar(results.keys(), results.values(), width=0.4, color="indianred")
    ax.bar_label(bars)
    plt.ylabel("Accuracy")

    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    plt.tight_layout()

    plt.savefig(ROOT_DIR + "/generated_result/accuracy.pdf")
    plt.show()

    print(results)


def extract_answer(dataset, text):
    if dataset == "commonsenseqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("A|B|C|D|E")][-1]
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "strategyqa":
        pred = text.lower()
        pred = re.sub("\"|'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    else:
        raise NotImplementedError(" not support dataset: {}".format(dataset))


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

    # read dataset file
    if dataset.lower() in ["strategyqa"]:
        with open(datapath) as f:
            if dataset.lower() in ["strategyqa"]:
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
                id = "temp_{}".format(idx)
                questions.append(q)
                answers.append(a)
                ids.append(id)
    elif dataset.lower() in ["commonsenseqa", "commonsenseqa_cot"]:
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

                id = "temp_{}".format(idx)
                questions.append(q)
                answers.append(a)
                ids.append(id)

                if dataset == "commonsenseqa_cot":
                    whole_a = json_res["answer"]
                    whole_answers.append(whole_a)

    if dataset == "commonsenseqa_cot":
        return questions, answers, ids, whole_answers
    return questions, answers, ids


def construct_input(prompt, text):
    inputs = "Q:" + text + "\nA: " + prompt
    return inputs


if __name__ == "__main__":
    text = "\n\nRelevant information: \nMona Lisa: Located in the Louvre Museum in Paris, France\nVenus de Milo: Located in the Louvre Museum in Paris, France\n\nPlan: \nStep 1: (A) Identify the locations of the Mona Lisa and the Venus de Milo\nStep 2: Compare the locations\n\nAnswer: \nYes, the Mona Lisa and the Venus de Milo are both located in the Louvre Museum in Paris, France."
    ans = extract_answer("commonsenseqa", text)
    print(ans)
