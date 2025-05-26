import re
import os
import time
import json
import base64
import random
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Optional, Tuple, Union


def create_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 10000,
    top_p: float = 1.0,
    temperature: float = 1.0,
    presence_penalty: float = 1.0,
    stream: bool = False,
) -> Any:
    """
    Creates a chat completion request.
    创建聊天请求。

    Parameters:
    - api_key (str): API key for authentication.
    - base_url (str): The base URL for the API endpoint.
    - model (str): model name.
    - messages (List[Dict[str, Any]]): messages list [{"role": "system", "content": "You are a helpful assistant."}].
    - max_tokens (int, optional): The maximum number of tokens to generate in the completion. Default is 10000.
    - top_p (float, optional): Controls nucleus sampling. Default is 1.0.
    - temperature (float, optional):  Sampling temperature. Higher values mean more random completions. Default is 1.0.
    - presence_penalty (float, optional):
    - stream (bool, optional): Whether to stream the response. Default is False.
    Returns:
    - Any: The response from the chat completion API.
    """

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            timeout=60,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            top_p=top_p,
        )
        if response:
            return response.choices[0].message.content.strip()
        else:
            print(f"OpenAI API returned no response: {response}")
            return None
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise e
