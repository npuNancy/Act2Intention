import re
import os
import time
import json
import random
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple, Union

from request_api import create_chat_completion

SYSTEM_PROMPT = """<Role>You need to act as a real user.</Role>  
<Task>Your task is to generate realistic app usage intention trajectories based on the user profile and scenario I provide.</Task>  
<Rule>  
1. Each intention must include a specific Time, a concrete App name, and a realistic Intention.  
2. Intentions should follow a logical flow and connect coherently.  
3. Ensure each intention contains only one brief and specific action within an app; avoid combining multiple actions in a single description.  
4. Objectively describe smartphone usage intentions without additional content [e.g., reasoning, interpretations, or subjective opinions].  
5. Ensure all mentioned apps and their corresponding operations are realistic and feasible.  
6. Strictly output in JSON format: List[{"Time": "string","APP": "string","Intention": "string"}]
</Rule>"""


class GenIntentTrajectory:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_iterations = 3
        self.intent_length_max = 100

        self.user_personas = "simulator/user_personas.json"
        self.save_dir = "simulator/trajectory"

    def is_valid_json(self, json_string):
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    def parse_content(self, input_string):
        input_string = input_string.replace("\n", "").strip()
        pattern = r"\{.*?\}"
        matches = re.findall(pattern, input_string, re.DOTALL)

        res = []
        if len(matches) != 0:
            for match in matches:
                if not self.is_valid_json(match):
                    continue
                json_data = json.loads(match)
                res.append(json_data)
        return res

    def generate_intentions(self, user_profile: str, scenario: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_profile} {scenario}"},
        ]

        all_intentions = []
        max_iterations = self.max_iterations
        while len(all_intentions) > self.intent_length_max or max_iterations > 0:
            max_iterations -= 1
            try:
                response = create_chat_completion(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                    messages=messages,
                )

                current_intentions = self.parse_content(response)
                all_intentions.extend(current_intentions)

                if not current_intentions:
                    print("Error: No intentions generated.")
                    print(f"{response=}")
                    break

                # update
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": 'You need to continue generating intent trajectories for subsequent time periods based on the generated intent trajectories.\nStrictly output in JSON format: List[{"Time": "string","APP": "string","Intention": "string"}]',
                    }
                )
            except Exception as e:
                print(f"Error generating intentions: {e}")
                break

        return all_intentions

    def run(self):
        with open(self.user_personas, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        for id, persona in user_profiles.items():
            gen_intentions = self.generate_intentions(persona, "")
            if not gen_intentions:
                print(f"Error: No intentions generated for persona {id}.")
                continue

            # Save to json
            output_file = os.path.join(self.save_dir, f"intentions_{id}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(gen_intentions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("API_HOST")
    MODEL = os.getenv("API_MODEL_NAME")

    if not API_KEY or not BASE_URL or not MODEL:
        raise ValueError("API_KEY, API_HOST, and API_MODEL_NAME must be set in the environment variables.")

    gen_intent_trajectory = GenIntentTrajectory(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    gen_intent_trajectory.run()
