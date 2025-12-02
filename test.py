# test.py (safe version)

import os
import requests

API_URL = "https://api-inference.huggingface.com/models/meta-llama/Meta-Llama-3-8B-Instruct"

HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY is None:
    raise RuntimeError("HF_API_KEY environment variable is not set.")

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload: dict) -> dict:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def main():
    prompt = "Hello! Please introduce yourself briefly."
    print("Sending prompt:", prompt)
    out = query({"inputs": prompt})
    print("Raw response:")
    print(out)

if __name__ == "__main__":
    main()
