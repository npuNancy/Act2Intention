import os
import re
import json
import numpy as np
from collections import defaultdict
from transformers.utils import is_nltk_available
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def get_event_list():
    dir1 = "data/trajectory/test/"
    dir2 = "data/trajectory/train"
    file_list1 = os.listdir(dir1)
    file_list2 = os.listdir(dir2)

    file_path1 = [os.path.join(dir1, filename) for filename in file_list1]
    file_path2 = [os.path.join(dir2, filename) for filename in file_list2]

    file_path = file_path1 + file_path2
    event_dict = defaultdict(int)
    for file in file_path:
        with open(file, "r", encoding="utf-8") as f:
            event_track_data = json.load(f)

        for item in event_track_data:
            event_dict[item["event"]] += 1
    return event_dict.keys()


def parse_content(input_string):
    """
    Description:
        parse Event and Behavior in content
    Args:
        content (str)
    Returns:
        operations_dict (dict):
        {
            "Event": "",
            "Behaviour": ""
        }
    """

    def is_valid_json(json_string):
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    pattern = r"\{.*?\}"
    matches = re.findall(pattern, input_string, re.DOTALL)

    res = {"Event": "", "Behaviour": ""}
    if len(matches) != 0 and is_valid_json(matches[0]):
        json_data = json.loads(matches[0])
        if "Event" in json_data:
            res["Event"] = json_data["Event"]
        elif "event" in json_data:
            res["Event"] = json_data["event"]

        if "Behaviour" in json_data:
            res["Behaviour"] = json_data["Behaviour"]
        elif "behaviour" in json_data:
            res["Behaviour"] = json_data["behaviour"]
        elif "Behavior" in json_data:
            res["Behaviour"] = json_data["Behavior"]
        elif "behavior" in json_data:
            res["Behaviour"] = json_data["behavior"]

    Event = res.get("Event", "")
    Behaviour = res.get("Behaviour", None)
    return Event, Behaviour


def evaluate_prediction(item):
    try:
        label = item["label"]
        predict = item["predict"]
    except Exception as e:
        print(str(e))

    event_true, behaviour_true = parse_content(label)
    event_pred, behaviour_pred = parse_content(predict)

    if behaviour_true is None or behaviour_pred is None:
        behaviour_blue4 = 0
    behaviour_true = behaviour_true.replace("<|im_start|>", "").replace("<|im_end|>", "")
    behaviour_pred = behaviour_pred.replace("<|im_start|>", "").replace("<|im_end|>", "")
    bleu_score = sentence_bleu([behaviour_true], behaviour_pred, smoothing_function=SmoothingFunction().method3)
    behaviour_blue4 = round(bleu_score * 100, 4)

    return event_true, event_pred, behaviour_blue4


def eval3(filepath):
    # read json file
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    true_list = []
    pred_list = []
    blue4_list = []
    for item in data:
        prediction = json.loads(item)

        # get Event acc and Behavior的Bleu-4
        event_true, event_pred, behaviour_blue4 = evaluate_prediction(prediction)
        blue4_list.append(behaviour_blue4)

        true_list.append(event_true)
        pred_list.append(event_pred)

    accuracy = accuracy_score(true_list, pred_list)
    precision = precision_score(true_list, pred_list, labels=classes, average="weighted", zero_division=0)
    recall = recall_score(true_list, pred_list, labels=classes, average="weighted", zero_division=0)
    f1_score = f1_score(true_list, pred_list, labels=classes, average="weighted", zero_division=0)

    blue4_mean = np.mean(blue4_list)

    accuracy = round(accuracy * 100, 2)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    f1_score = round(f1_score * 100, 2)
    blue4_mean = round(blue4_mean, 2)

    print(f"{accuracy=}")
    print(f"{precision=}")
    print(f"{recall=}")
    print(f"{f1_score=}")
    print(f"{blue4_mean=}")


classes = get_event_list()
if __name__ == "__main__":

    eval3("path/to/stage3_result.jsonl")
