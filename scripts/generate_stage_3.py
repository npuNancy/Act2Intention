import os
import sys
import json
import math
import random
from enum import Enum
from itertools import combinations
from tqdm import tqdm


STAGE_3_SYSTEM_PROMPT = """<Role> You are a helpful assistant that provides proactive suggestions to the user. </Role>
<Task> Understand what the user is doing and predict their next behavior based on historical behaviors.</Task>
<Format> 
- Strictly respond in the following JSON format:
{"Event": "EVENT class to which behavior belongs", "Behaviour": "Describe the predicted behavior."}
- DO NOT output anything other than JSON.
</Format>
<Rules>
- Ensure the predicted behavior is relevant to the historical behaviors. 
- Focus on the user's current needs and predict helpful behaviors.
- Consider the timing of Behaviour and the EVNET classes.
</Rules>"""


class GenStage3Dataset:
    def __init__(self, event_track_data_dir, save_basedir, split="train", window_size=51, repeat=7):

        self.way_20 = [
            "E-commerce platform",
            "Community",
            "Message",
            "Take photo",
            "Chatting",
            "Watch news",
            "Check weather",
            "Edit media",
            "Read e-books",
            "Set Alarm",
            "Office chat",
            "Listen music",
            "Check map",
            "Knowledge acquisition",
            "Financial management",
            "Gaming community",
            "Audiobook",
            "Order takeout",
            "Buy ticket",
            "Take notes",
        ]
        self.event_track_data_dir = event_track_data_dir
        self.save_basedir = save_basedir
        self.window_size = window_size
        self.repeat = repeat
        self.split = split

        choices = '"' + '"\n"'.join(self.way_20) + '"'
        system_prompt_append = f"<choices> When proposing a predicted behavior, the EVENT can be only selected from the following options:\n{choices}\n</choices>"
        self.system_prompt = STAGE_3_SYSTEM_PROMPT + system_prompt_append

        self.run()

    def split_array_into_segments(self, arr: list) -> list[list]:
        repeat = self.repeat
        window_size = self.window_size
        step = window_size
        assert repeat > 0

        end_init_offset_list = random.choices(range(window_size, window_size * 2), k=repeat)

        segments = []
        for idx in range(repeat):
            end = window_size + end_init_offset_list[idx]
            while end < len(arr):
                start = end - window_size
                segment = arr[start:end]
                segments.append(segment)
                end += step

        return segments

    def generate_dataset(self, event_track_data: list) -> list:
        n_way = self.way_20
        if len(event_track_data) == 0:
            return []
        result_list = []

        # split json_data
        segments = self.split_array_into_segments(event_track_data)

        for segment in segments:
            if segment[-1]["event"] not in n_way:
                continue

            input_dict = {
                "Instructions": "Now analyze the history behaviour and provide a task if you think the user needs your help.",
                "Observations": [
                    {
                        "Step_id": i + 1,
                        "Time": item["datetime"].split(".")[0],
                        "Event": item["event"],
                        "Behaviour": item["behavior"],
                    }
                    for i, item in enumerate(segment[:-1])
                ],
            }

            output_dict = {"Event": segment[-1]["event"], "Behaviour": segment[-1]["behavior"]}

            my_conversation = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(input_dict, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(output_dict, ensure_ascii=False)},
                ]
            }
            result_list.append(my_conversation)
        return result_list

    def run(self):
        file_list = os.listdir(self.event_track_data_dir)

        datasets = []

        for filename in tqdm(file_list):
            if not filename.endswith(".json"):
                continue

            # read json file
            filepath = os.path.join(self.event_track_data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                event_track_data = json.load(f)

            datasets.extend(self.generate_dataset(event_track_data))

        random.shuffle(datasets)

        # save
        os.makedirs(self.save_basedir, exist_ok=True)
        save_path = os.path.join(self.save_basedir, f"{self.split}_len_{self.window_size-1}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(datasets, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    save_basedir = "data/datasets/stage3/"
    gen_stage_3 = GenStage3Dataset("data/trajectory/test", save_basedir, split="test")
    gen_stage_3 = GenStage3Dataset("data/trajectory/train", save_basedir, split="train")
