import os

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import wrappers
from dotenv import load_dotenv

load_dotenv()


def _langsmith_tracing_enabled():
    value = os.getenv("LANGSMITH_TRACING_V2", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _create_gemini_client(api_key):
    client = genai.Client(api_key=api_key)
    if _langsmith_tracing_enabled():
        return wrappers.wrap_gemini(client)
    return client


def _gemini_generate_text(api_key, model_name, prompt):
    """Generate plain text from Gemini using the google.genai SDK."""
    client = _create_gemini_client(api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text

    # Fallback for responses where text is segmented into parts.
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts)