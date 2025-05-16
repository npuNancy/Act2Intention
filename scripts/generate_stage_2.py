import os
import sys
import json
import math
import random
from enum import Enum
from itertools import combinations
from tqdm import tqdm


STAGE_2_SYSTEM_PROMPT = """<Role>You are a mobile action descriptions analysis expert, responsible for identifying user behaviors on mobile devices. </Role><Task> You are tasked with grouping action description sequences and identifying the corresponding behavior for each group. You need output in JSON format.</Task>
Please process user input according to the following rules:
<Rules>
1. Behavior Segmentation Analysis:
    - Identify the boundaries between different behaviors in a continuous action descriptions sequence.
    - Determine behavior switches based on user intent, application scenarios, and temporal continuity.
    - Each independent behavior must contain at least ONE action descriptions.
2. Behavior Description Standards
    - Use a verb-object structure (verb + target object) (e.g., "Modify system settings")
    - Include core verbs and application scenarios
    - Keep within 20 English words
    - Avoid using the exact wording from the action description steps
3. Formatting Requirements:
    - Strictly output in JSON format: {string: List[int]}.
    - The output is a dictionary with Behavior Descriptions as keys and a list of Action Description INDICES as values.
    - Each Behavior Description is a list of action description indices.
    - Index starts at 1.
</Rules>
<Format> Strictly output in JSON format: {"Behavior Description": [steps]}! </Format>
<Example> ### Demo Example (Do not use directly) ###
<Input>
Step 1: Open settings
Step 2: Adjust volume
Step 3: Return to home screen
Step 4: Launch camera
Step 5: Switch to night mode
</Input><Output>{"System settings adjustment": [1,2,3], "Shooting mode configuration": [4,5]}</Output></Example>

### Steps for Processing New Input ###
1. Analyze the contextual relevance of the Action Description sequence
2. Identify the points of user intent transition
5. Output format strictly follows the requirements"""


def get_trajectory(file_path_list):
    for file in tqdm(file_path_list):
        if not file.endswith(".json"):
            continue

        with open(file, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

        yield trajectory


def generate_dataset_json_format(trajectory, segment_length):
    """
    Desription:
        Use sliding window to split the trajectory into multiple segments
    Args:
        trajectory (dict): user trajectory
        segment_length (int): length of the sliding window
    Returns:
        dataset (list)
    """

    window = segment_length
    step = max(1, window // 2)

    split_lists = []
    for i in range(0, len(trajectory) - window + 1, step):
        split_lists.append(trajectory[i : i + window])

    result_list = []
    for segment in split_lists:
        ## user input
        action_description_list = [step["action_description"] for behavior in segment for step in behavior["steps"]]
        action_description_list = [
            f"step_{i+1}: {action_description}" for i, action_description in enumerate(action_description_list)
        ]
        action_description_string = "\n".join(action_description_list)

        ### assistant output
        output_dict = {}
        step_start = 1
        for behavior in segment:
            output_dict[behavior["behavior"]] = [step_start + i for i in range(len(behavior["steps"]))]
            step_start += len(behavior["steps"])

        my_conversation = {
            "messages": [
                {"role": "system", "content": STAGE_2_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Please handle the following sequence of action descriptions:\n{action_description_string}\n",
                },
                {"role": "assistant", "content": json.dumps(output_dict, ensure_ascii=False)},
            ],
        }
        result_list.append(my_conversation)

    return result_list


def generate_stage_2(
    file_path_list,
    save_dir,
    split="train",
    segment_length=5,
):
    dataset_list = []
    for trajectory in tqdm(get_trajectory(file_path_list)):
        dataset_gen = generate_dataset_json_format(trajectory, segment_length)
        dataset_list.append(dataset_gen)

    # save
    os.makedirs(save_dir, exist_ok=True)
    savepath = os.path.join(save_dir, f"{split}_len_{segment_length}.json")
    with open(savepath, "w", encoding="utf-8") as f:
        random.shuffle(dataset_list)
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    dir_test = "data/trajectory/test"
    dir_train = "data/trajectory/train"

    file_list_test = os.listdir(dir_test)
    file_list_train = os.listdir(dir_train)

    file_path_list_test = [os.path.join(dir_test, filename) for filename in file_list_test]
    file_path_list_train = [os.path.join(dir_train, filename) for filename in file_list_train]

    generate_stage_2(file_path_list_test, save_dir="data/datasets/stage2/", split="test")
    generate_stage_2(file_path_list_train, save_dir="data/datasets/stage2/", split="train")
