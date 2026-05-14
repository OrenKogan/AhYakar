"""
Agent 3 — Appointment Planner
==============================
Composes a warm, human-readable appointment proposal and asks the patient
for confirmation before anything is booked.

Only invoked when the Medical Advisor's action is 'schedule_gp' or
'schedule_specialist'.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are an Appointment Scheduling agent for a medical AI.

You are triggered only when a doctor visit is required. Your job is simple:

1. State in ONE sentence that a visit to the relevant specialist is needed.
2. Ask exactly this question (fill in the specialist): "Would you like me to help you schedule an appointment to the [Specialist]?"

No extra sentences. No explanations. No filler. No reassurances.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "proposal_text": "One sentence: visit is needed. Then: 'Would you like me to help you schedule an appointment to the [Specialist]?'",
  "appointment_details": {
    "type": "gp | specialist",
    "specialist_type": "e.g. Family Doctor, Cardiologist",
    "urgency": "immediately | within_24h | this_week | whenever",
    "reason": "diagnosis in 1 sentence"
  }
}
"""


def plan(client: OpenAI, triage: dict, advice: dict, location: dict | None = None) -> dict:
    """
    Run the Appointment Planner agent.

    Args:
        client : OpenAI-compatible client.
        triage : Triage dict from Agent 1.
        advice : Advice dict from Agent 2.

    Returns:
        A dict with 'proposal_text' and 'appointment_details'.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Triage report:\n{json.dumps(triage, indent=2)}\n\n"
                f"Medical advice:\n{json.dumps(advice, indent=2)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-OpenRouter-Title": "AhYakar - Appointment Planner",
        },
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        specialist = advice.get("specialist_type", "Family Doctor")
        urgency = advice.get("urgency", "this_week")
        return {
            "proposal_text": (
                f"Based on your symptoms, I recommend scheduling an appointment with a "
                f"{specialist} ({urgency.replace('_', ' ')}). "
                "Would you like me to book this for you?"
            ),
            "appointment_details": {
                "type": advice.get("action", "schedule_gp").replace("schedule_", ""),
                "specialist_type": specialist,
                "urgency": urgency,
                "reason": triage.get("reasoning", "Medical evaluation recommended"),
            },
        }
