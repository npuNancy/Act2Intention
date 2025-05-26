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


class GenPersona:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dir = "simulator/intentions"
        self.save_file = "simulator/user_personas.json"

        self.SYSTEM_PROMPT = """<Role>You are a user data analyst</Role>
<Task>Your task is to generate a comprehensive and personalized user profile description based on the user's historical mobile phone usage intent trajectory.</Task>
<Rule>  
0. The input format is: List[(Time, APP, Intention, Event)], where "Event" represents the category of each user intent.  
1. First, you should analyze the intentions and their corresponding event categories to determine which events are semantically similar.  
2. Based on a clear understanding of the event semantics, merge high-frequency sub-items according to their semantic similarity. The merging process must adhere to the following requirements:  
    (1) Carefully consider whether the semantics of the sub-items to be merged are truly identical and whether there is a better way to merge them.  
    (2) Each merge will result in slightly more generalized semantics. Pay attention to the number of merges:  
        - Merging too much will make the semantics overly broad and vague, applicable to most users, and failing to reflect the individual's unique traits.  
        - Merging too little will make the behavior patterns overly fragmented and detailed, obscuring the user's core behavior patterns.  
    (3) Before each merge, you may refer to the timestamps of the sub-items in the historical behavior sequence to better inform your judgment. Additionally, if a behavior pattern exhibits clear temporal characteristics, state them directly.  
3. Finally, based on the historical behavior sequence and your merged behavior patterns, provide a thorough analytical summary of the user's profile. The summary should be comprehensive while highlighting the user's personalized traits.  
4. Maintain an objective description and avoid excessive speculation. All inferences should be grounded in the user's actual historical intents and behavior patterns. Rather than describing highly uncertain imagined content, focus only on what can be confidently deduced from the user's historical intent sequence and behavior patterns. Minimize aesthetic descriptions as much as possible.  
5. All the Intentions mentioned above are collected from users' mobile phone usage.
6. Strictly output in JSON format: {"Persona": str}, and DO NOT output anything other than JSON.
</Rule>"""

    def is_valid_json(self, json_string):
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    def parse_content(self, input_string):

        pattern = r"\{.*?\}"
        matches = re.findall(pattern, input_string, re.DOTALL)

        if len(matches) != 0 and self.is_valid_json(matches[0]):
            json_data = json.loads(matches[0])

            if "Persona" in json_data:
                return json_data["Persona"]

        return None

    def generate_persona(self, data: List[Tuple[str, str, str, str]]) -> Dict[str, str]:
        """
        Generates a user persona based on the provided data.
        :param data: List of tuples containing (Time, APP, Intention, Event)
        :return: Generated persona in JSON format
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(data)},
        ]
        response = create_chat_completion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
        )
        return response

    def run(self):
        persona_res_dict = {}

        file_list = os.listdir(self.dir)
        for file_idx, file in enumerate(file_list):
            with open(os.path.join(self.dir, file), "r", encoding="utf-8") as f:
                data_csv = f.readlines()[1:-1]
            data = [f"({x.strip()})" for x in data_csv]

            response = self.generate_persona(data)
            persona = self.parse_content(response)
            persona_res_dict[file_idx] = persona

        ## save personas
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump(persona_res_dict, f, ensure_ascii=False, indent=4)


class GenPersonaWithoutIntent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.N = 10
        self.save_file = "simulator/user_personas.json"

        self.SYSTEM_PROMPT = """<Role>You are a user data analyst</Role>
<Task>Your task is to generate a comprehensive and personalized user profile description based on the user's historical mobile phone usage intent trajectory.</Task>
<Rule>  
0. The input format is: List[(Time, APP, Intention, Event)], where "Event" represents the category of each user intent.  
1. First, you should analyze the intentions and their corresponding event categories to determine which events are semantically similar.  
2. Based on a clear understanding of the event semantics, merge high-frequency sub-items according to their semantic similarity. The merging process must adhere to the following requirements:  
    (1) Carefully consider whether the semantics of the sub-items to be merged are truly identical and whether there is a better way to merge them.  
    (2) Each merge will result in slightly more generalized semantics. Pay attention to the number of merges:  
        - Merging too much will make the semantics overly broad and vague, applicable to most users, and failing to reflect the individual's unique traits.  
        - Merging too little will make the behavior patterns overly fragmented and detailed, obscuring the user's core behavior patterns.  
    (3) Before each merge, you may refer to the timestamps of the sub-items in the historical behavior sequence to better inform your judgment. Additionally, if a behavior pattern exhibits clear temporal characteristics, state them directly.  
3. Finally, based on the historical behavior sequence and your merged behavior patterns, provide a thorough analytical summary of the user's profile. The summary should be comprehensive while highlighting the user's personalized traits.  
4. Maintain an objective description and avoid excessive speculation. All inferences should be grounded in the user's actual historical intents and behavior patterns. Rather than describing highly uncertain imagined content, focus only on what can be confidently deduced from the user's historical intent sequence and behavior patterns. Minimize aesthetic descriptions as much as possible.  
5. All the Intentions mentioned above are collected from users' mobile phone usage.
6. Strictly output in JSON format: {"Persona": str}, and DO NOT output anything other than JSON.
</Rule>"""

    def is_valid_json(self, json_string):
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    def parse_content(self, input_string):

        pattern = r"\{.*?\}"
        matches = re.findall(pattern, input_string, re.DOTALL)

        if len(matches) != 0 and self.is_valid_json(matches[0]):
            json_data = json.loads(matches[0])

            if "Persona" in json_data:
                return json_data["Persona"]

        return None

    def generate_persona(self) -> Dict[str, str]:
        """
        Generates a user persona based on the provided data.
        :param data: List of tuples containing (Time, APP, Intention, Event)
        :return: Generated persona in JSON format
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "generate a persona"},
        ]
        response = create_chat_completion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
        )
        return response

    def run(self):
        persona_res_dict = {}

        for i in range(self.N):
            response = self.generate_persona()
            persona = self.parse_content(response)
            persona_res_dict[i] = persona

        ## save personas
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump(persona_res_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("API_HOST")
    MODEL = os.getenv("API_MODEL_NAME")
    if not API_KEY or not BASE_URL or not MODEL:
        raise ValueError("API_KEY, API_HOST, and API_MODEL_NAME must be set in the environment variables.")

    """
    For privacy reasons, here we only provide one example.
    """
    gen_persona = GenPersona(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    gen_persona.run()

"""

Generate a detailed user persona focused on Android smartphone users, analyzing intent-driven behavioral patterns and contextual correlations. Structure the output strictly using these markdown headers and subsections:  
# Profile #
## Gender  ##
## Age range  ##
## Occupation  ##
## City tier  ##
## Monthly income range  ##
## Education level  ##
## Core Characteristics  ##
## Interests & Hobbies  ##
## Primary leisure activities  ##

# Content consumption preferences (video/article/gaming)  #
## Brand affinities  ##
## Consumption Habits  ##
## Online/offline spending ratio  ##
## Preferred payment methods  ##
## Impulse vs planned purchasing triggers  ##


# Frequently Used Apps  #
## Top 5 daily Android apps (excluding system apps)   ##
## App category clusters (e.g., shopping/social/fintech)   ##
## Pre-installed vs downloaded app patterns   ##


# Browsing Behavior  #
## Primary content discovery pathways  ##
## Notification-triggered vs intentional browsing  ##


## Cross-app content consumption flow  
## Search Behavior  ##
## Common search verticals (products/services/info)  ##


Voice vs text search frequency  
Google Search vs in-app search patterns  

Social Context  
Active Hours  
Peak engagement periods mapped to:  
Workday vs weekend patterns  


App-specific usage spikes  


Social Interaction Patterns  
Dominant communication channels (WhatsApp/Telegram/etc.)  
Group chat participation frequency  


Social media cross-posting habits  


Behavioral Analysis Requirements:  
Map 3 intent chains showing contextual transitions:  


Trigger (e.g., push notification) → Action → Next Intent → Outcome  
Highlight Android-specific behaviors:  


Customization habits (launchers/widgets)  


Google Play Store vs sideloaded app usage  


OS version impact on feature adoption  
Identify context bridges between:  


Search → Shopping apps  


Social media → Browser transitions  


Format Rules:  
Use bullet points under each header  


Bold key behavioral differentiators (e.g., Prefers Google Pay over UPI)  


Include emojis to represent common app categories (🛍️📱💬)  


Add timeline diagrams for daily active hours (🕘→🕛→🕠)  


Provide actionable insights for Android app developers under each behavior pattern section.




"""
