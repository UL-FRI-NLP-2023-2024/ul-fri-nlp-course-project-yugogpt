from prompt_creator import PromptCreator
from utils_funs import calculate_accuracy, construct_input, load_data

if __name__ == "__main__":
    # # define dataset
    # dataset_name = "commonsenseqa"  # "strategyqa"

    # # define strategy
    # strategy = "zero_shot"  # "plan_and_solve", "zero_shot"

    # # load data
    # q, a, ids = load_data(dataset_name)

    # # create prompt
    # pc = PromptCreator(strategy, (q, a, ids))
    # prompt = pc.get_next_prompt()

    # # create input
    # text = "What is the answer to this question?"
    # inputs = construct_input(prompt, text)

    # print(inputs)
    calculate_accuracy()
