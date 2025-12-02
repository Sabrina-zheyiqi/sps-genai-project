import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from the project root ".env".
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

if not HF_API_KEY:
    raise ValueError("HF_API_KEY is not set in environment variables or .env")

# Global client instance, lazily initialized.
_client: Optional[InferenceClient] = None


def get_client() -> InferenceClient:
    """
    Lazily initialize and return a global InferenceClient.

    This client uses the Hugging Face Inference API under the hood and
    will automatically route calls for the given model ID.
    """
    global _client
    if _client is None:
        _client = InferenceClient(model=HF_MODEL_ID, token=HF_API_KEY)
    return _client


def call_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Call a conversational LLM (e.g., Llama 3 8B Instruct) using the
    chat_completion API. We send a simple system+user message and
    return the assistant's reply as a plain string.

    max_tokens:
        Maximum number of tokens to generate for this call.
    temperature:
        Sampling temperature used by the model.
    """
    client = get_client()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, cautious medical-style assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    resp = client.chat_completion(
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
    )

    # Extract the content from the first choice.
    choice = resp.choices[0]
    msg = choice.message

    # msg may be a dict or an object depending on huggingface_hub version.
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")

    return content
