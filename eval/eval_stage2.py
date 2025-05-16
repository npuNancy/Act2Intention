import os
import re
import json
import numpy as np
from typing import List, Dict
from datetime import datetime


from transformers.utils import is_nltk_available

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def cal_behaviours_bleu(behaviours_pred: List[str], behaviours_true: List[str]) -> float:
    min_len = min(len(behaviours_pred), len(behaviours_true))

    try:
        if min_len == 0:
            return 0.0

        # remove <|im_start|> 和 <|im_end|>
        behaviours_pred = [
            behaviour.replace("<|im_start|>", "").replace("<|im_end|>", "") for behaviour in behaviours_pred
        ]
        behaviours_true = [
            behaviour.replace("<|im_start|>", "").replace("<|im_end|>", "") for behaviour in behaviours_true
        ]

        bleu_4 = []
        for pred, label in zip(behaviours_pred, behaviours_true):
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            bleu_4.append(round(bleu_score, 4))

        result = np.mean(bleu_4)
        return result
    except Exception as e:
        print(e)


def parse_content(content):
    """
    Description: parse the behavior and action in content
    Args:
        content (str): JSON string
    Returns:
        operations_dict (dict):
        {
            "behavior_1": [1,2,3],
            "behavior_2": [4,5,6,7]
        }
    """
    content = content.replace("\n", "").strip()
    pattern = re.compile(r"\"(.*?)\"\s*:\s*\[(.*?)\]")
    matches = pattern.findall(content)
    if len(matches) == 0:
        print(f"no thing: {content}")
        return {}

    result = {}
    for match in matches:
        try:
            result[match[0]] = [x.replace("[", "").replace("]", "").strip() for x in match[1].split(",")]
        except Exception as e:
            print(f"Error: {e}")
            result[match[0]] = []

    return result


def evaluate_prediction(item):
    try:
        label = item["label"]
        predict = item["predict"]
    except Exception as e:
        print(e)

    # extra json str {"xxxx":[1,2,3]}
    dict_true = parse_content(label)
    dict_pred = parse_content(predict)

    group_num_val = len(dict_true.keys()) == len(dict_pred.keys())

    if group_num_val == True:
        group_location_sum = 0  # acc
        operations_dict_zipped = zip(dict_true.values(), dict_pred.values())
        for sub_seq_pred, sub_seq_true in operations_dict_zipped:
            group_location_sum += 1 if len(sub_seq_pred) == len(sub_seq_true) else 0
        group_location_val = group_location_sum / len(dict_true.values())
    else:
        group_location_val = 0

    behaviours_pred, behaviours_true = dict_pred.keys(), dict_true.keys()
    behaviours_blue4 = cal_behaviours_bleu(behaviours_pred, behaviours_true)

    return {
        "group_num_val": group_num_val,
        "group_location_val": group_location_val,
        "behaviours_blue4": behaviours_blue4,
        "group_num_true": len(behaviours_true),
        "group_num_pred": len(behaviours_pred),
    }


def eval_2(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    group_num_val_list = []
    group_location_val_list = []
    behaviours_blue4_list = []

    group_num_true_list = []  # true length
    group_num_pred_list = []  # pred length

    for item in data:
        prediction = json.loads(item)
        result = evaluate_prediction(prediction)
        group_num_val_list.append(result["group_num_val"])
        group_location_val_list.append(result["group_location_val"])
        behaviours_blue4_list.append(result["behaviours_blue4"])
        group_num_true_list.append(result["group_num_true"])
        group_num_pred_list.append(result["group_num_pred"])

    # res
    group_num_val = np.mean(group_num_val_list) * 100
    group_location_val = np.mean(group_location_val_list) * 100
    behaviours_blue4 = np.mean(behaviours_blue4_list) * 100

    print(f"{group_num_val=}")
    print(f"{group_location_val=}")
    print(f"{behaviours_blue4=}")


if __name__ == "__main__":
    eval_2("path/to/stage2_result.jsonl")
