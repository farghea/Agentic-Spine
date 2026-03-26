from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI


def _gemini_generate_text(api_key, model_name, prompt):
    """Generate plain text from Gemini using the google.genai SDK."""
    client = genai.Client(api_key=api_key)
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