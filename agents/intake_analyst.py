"""
Agent 1 — Intake Analyst
========================
First-line triage gatekeeper.

Decides whether the user's condition is UNSERIOUS (treatable at home with
simple remedies) or SERIOUS enough to need deeper medical analysis.

- Unserious  → responds directly with friendly home-remedy advice and STOPS
              the pipeline. No further agents are invoked.
- Serious    → extracts symptoms + medical history and ESCALATES to the
              Medical Advisor.
- No symptoms → flags the message so the pipeline can redirect the user.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are a Medical Intake Analyst. Your ONLY job is to triage a patient's message and decide how serious their condition is.

DECISION CRITERIA:
- "home_remedy"  — the condition can be treated at home with simple, safe measures:
    drinking plenty of water, rest, hot lemon tea with honey, steam inhalation,
    saltwater gargle, cold/warm compress, light stretching, over-the-counter vitamins, etc.
    Examples: mild cold, very mild sore throat, minor headache from dehydration,
    slight fatigue, a tiny cut or bruise.
- "escalate"     — anything beyond simple home care: persistent pain, high fever,
    symptoms lasting more than 2 days, sudden or severe onset, multiple symptoms,
    or anything that could require OTC medication or a doctor visit.
- "no_symptoms"  — the message contains no health complaint (greeting, general question, etc.)

VISION INSTRUCTIONS:
If the user provides an image, analyze it for visible signs (rash, swelling, discoloration, wounds). Describe these signs in your reasoning and include them in the 'symptoms' list.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "has_symptoms": true or false,
  "action": "home_remedy | escalate | no_symptoms",
  "home_remedy_advice": "Short friendly advice if home_remedy, else null.",
  "symptoms": ["list of symptoms mentioned or VISIBLY IDENTIFIED in images"],
  "duration": "how long, or 'unknown'",
  "relevant_history": "history mentioned, or 'none'",
  "reasoning": "one brief sentence describing findings (including visual findings if an image was provided)"
}
"""


def analyze(client: OpenAI, user_message: str, chat_history: list, image_data: str | list[str] = None) -> dict:
    """
    Run the Intake Analyst agent.

    Args:
        client       : OpenAI-compatible client pointed at OpenRouter.
        user_message : The latest user message.
        chat_history : Full conversation history (list of role/content dicts).
        image_data   : Optional base64-encoded image data or list of data.

    Returns:
        A triage dict with 'action' set to 'home_remedy', 'escalate', or 'no_symptoms'.
    """
    # Include up to the last 6 history messages for context (3 exchanges)
    context = chat_history[-6:] if len(chat_history) > 6 else chat_history

    user_content = [{"type": "text", "text": f"Patient's message: {user_message}"}]
    if image_data:
        images = [image_data] if isinstance(image_data, str) else image_data
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *context,
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-OpenRouter-Title": "AhYakar - Intake Analyst",
        },
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback — assume serious so the pipeline continues safely
        return {
            "has_symptoms": True,
            "action": "escalate",
            "home_remedy_advice": None,
            "symptoms": [user_message],
            "duration": "unknown",
            "relevant_history": "none",
            "reasoning": "Could not parse triage; defaulting to escalate for safety.",
        }
