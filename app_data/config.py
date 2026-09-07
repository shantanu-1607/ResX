import os
from pathlib import Path
import logging
import sys

from dotenv import find_dotenv, load_dotenv

# Load .env from app_data, project root, or the current working directory.
env_app_data = Path(__file__).resolve().parent / ".env"
env_root = Path(__file__).resolve().parent.parent / ".env"

if env_app_data.exists():
    load_dotenv(env_app_data)
elif env_root.exists():
    load_dotenv(env_root)
else:
    load_dotenv(find_dotenv(usecwd=True))

groq_api = os.getenv("GROQ_API_KEY")
tavily_api = os.getenv("TAVILY_API_KEY")


logger = logging.getLogger(__name__)

# Validate required API keys
if not groq_api:
    logger.error("GROQ_API_KEY is required but not set. Please configure it in your .env file.")
    sys.exit(1)

if not tavily_api:
    logger.warning(
        "TAVILY_API_KEY is not set. Web search functionality will fail if requested. "
        "Set TAVILY_API_KEY in your .env file to enable web search."
    )

from groq import APIStatusError, Groq, RateLimitError

client = Groq(api_key=groq_api)


def llm_with_fallback(
    messages,
    primary_model,
    fallback_model=None,
    reasoning_effort=None,
    max_tokens=1200,
    fallback_max_tokens=None,
):
    kwargs = {
        "model": primary_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    return client.chat.completions.create(**kwargs)
