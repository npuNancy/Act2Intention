import os
import json
import random
import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

STAGE_1_SYSTEM_PROMPT = """
<Role>You are a professional GUI action Describer</Role>
<Task>You are given an action and the split-screenshot image (Before-action left, After-action right) on the mobile phone. Then, you need to describe the action.</Task>
<Rule>
- For click actions, a high-contrast red marker (white-bordered circle) shows the precise click location, with a green square surrounding it and a ’C’ label at the top-right corner of the square indicating the click.
- Strictly output in JSON format: {"Action_Description": str}
- The "Action_Description" field in this exact format: "[On/In] [PackageName], [Action Details], to [Purpose]"
- The "Action_Description" field needs to keep within 20 English words
- DO NOT output anything other than JSON
</Rule>
"""


def mark_click(image_path, x, y):
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)

    square_size = max(40, min(width, height) // 20)
    circle_radius = 10
    border_width = 5

    square_box = (x - square_size // 2, y - square_size // 2, x + square_size // 2, y + square_size // 2)
    draw.rectangle(square_box, outline="green", width=2)

    draw.ellipse(
        (
            x - circle_radius - border_width,
            y - circle_radius - border_width,
            x + circle_radius + border_width,
            y + circle_radius + border_width,
        ),
        fill="white",
    )
    draw.ellipse((x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius), fill="red")

    font = ImageFont.truetype("arial.ttf", 14)
    text_x = square_box[2] - 15
    text_y = square_box[1] + 2
    draw.text((text_x, text_y), "C", fill="green", font=font)

    return img


def merge_screenshots(left_path, right_path, action, i):
    save_name = f"{left_path.split('-')[0]}-{i+1}-to-{i+2}.png"
    save_path = os.path.join(base_dir, merge_image_base_dir, save_name)

    if os.path.exists(save_path):
        return save_path
    left_path = os.path.join(image_base_dir, left_path)
    right_path = os.path.join(image_base_dir, right_path)

    if action.startswith("CLICK"):
        _, coords = action.strip("]").split("[")
        x, y = map(int, coords.split(","))
        left_img = mark_click(left_path, x, y)
    else:
        left_img = Image.open(left_path)

    right_img = Image.open(right_path)

    gap = 10
    width = left_img.width + right_img.width + gap
    height = max(left_img.height, right_img.height)
    merged = Image.new("RGB", (width, height), "white")

    gradient = np.linspace(0, 255, gap // 2)
    gradient = np.concatenate([gradient, gradient[::-1]])
    gradient = np.tile(gradient, (left_img.height, 1)).reshape(left_img.height, gap)

    merged.paste(Image.fromarray(gradient), (left_img.width, 0))
    merged.paste(left_img, (0, 0))
    merged.paste(right_img, (left_img.width + gap, 0))

    # save
    merged.save(save_path)
    return save_path


def generate_conversation(steps, app):
    res = []
    for i in range(len(steps) - 1):
        step_obj = steps[i]
        action = step_obj["action"]
        action_description = step_obj["action_description"]
        img_path_before = step_obj["image_path"]
        img_path_after = steps[i + 1]["image_path"]

        merge_image_path = merge_screenshots(img_path_before, img_path_after, action, i)

        res.append(
            {
                "messages": [
                    {"role": "system", "content": STAGE_1_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The Action is: {action}, current app is: {app}."},
                            {"type": "image_url", "image_url": {"url": merge_image_path}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"Action_Description": action_description}, ensure_ascii=False),
                    },
                ],
            }
        )
    return res


base_dir = "/path/to/root"  # TODO

image_base_dir = "data/screenshots/screenshots"
merge_image_base_dir = "data/datasets/stage1/merge_images"

trajectory_dir_test = "data/trajectory/test"
file_list_test = os.listdir(trajectory_dir_test)
file_path_test = [os.path.join(trajectory_dir_test, filename) for filename in file_list_test]


res_conversations_list = []
behavior_actions = defaultdict(int)
for idx, file in enumerate(file_path_test):
    if not file.endswith(".json"):
        continue
    print(f"{'='*10}{idx}{'='*10}")
    with open(file, "r") as f:
        event_track_data = json.load(f)
    for item in tqdm(event_track_data):
        behavior = item["behavior"]

        if behavior in behavior_actions:
            continue

        behavior_actions[behavior] = 1  # mark

        steps = item["steps"]
        app = item["app"]

        res_conversations_list.extend(generate_conversation(steps, app))

# save
os.makedirs("data/datasets/stage1/", exist_ok=True)
with open("data/datasets/stage1/test.json", "w") as f:
    json.dump(res_conversations_list, f, ensure_ascii=False, indent=2)
