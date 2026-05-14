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

SYSTEM_PROMPT = """You are an empathetic Appointment Planning assistant for a medical AI.

You receive a triage report and medical advice, and you must compose a friendly, clear appointment proposal to present to the patient.

The patient must explicitly confirm before any appointment is booked — your job is to ask for that confirmation.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation — using this exact schema:
{
  "proposal_text": "A warm, clear message that: (1) briefly explains why an appointment is recommended, (2) states what type of doctor and urgency level, (3) politely asks whether the patient would like you to schedule it.",
  "appointment_details": {
    "type": "gp | specialist",
    "specialist_type": "e.g. General Practitioner, Cardiologist",
    "urgency": "immediately | within_24h | this_week | whenever",
    "reason": "brief medical reason (1 sentence)"
  }
}

Keep the proposal_text concise (3–4 sentences max). Be warm but professional.
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
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        specialist = advice.get("specialist_type", "General Practitioner")
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
