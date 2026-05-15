"""
Agent 2 — Medical Advisor
=========================
Deep clinical analysis agent. Only invoked when the Intake Analyst has
determined that the patient's condition is too serious for simple home remedies.

Responsibilities:
1. Fully analyze the symptoms and medical history.
2. Form a probable diagnosis.
3. Recommend OTC medication if appropriate (with dosage), always.
4. If a doctor visit is still needed: state the required specialty and ask
   the patient whether to go ahead and book an appointment.
5. In emergencies: direct to emergency services immediately.

The 'reply' field is the COMPLETE message shown to the patient — it
includes diagnosis, any OTC advice, and (if doctor_needed) the question
asking whether to proceed with booking.
"""

import json
from openai import OpenAI
from .utils import parse_json_response
from .rag_engine import retrieve_medical_context

SYSTEM_PROMPT = """You are a senior Medical Advisor AI. Answer directly, concisely, and without evasion.

YOUR KNOWLEDGE SOURCE:
You are provided with "Verified Medical Reference Data" for the most relevant conditions. You MUST prioritize the OTC dosages and self-care steps found in this data if it matches the patient's symptoms.

RULES:
- Direct answers only. No filler.
- State the most probable diagnosis (1 sentence).
- Always tell the patient WHAT TO DO: recommend a MAXIMUM of 1 or 2 essential OTC medications with name + dosage, and any relevant self-care steps (rest, hydration, diet, etc.). Do NOT overwhelm the patient with too many medications.
- IMPORTANT: Always use common ISRAELI brand names for medications (e.g., recommend 'Acamol' or 'Dexamol' instead of Acetaminophen, 'Nurofen' or 'Advil' instead of Ibuprofen). This is critical for our local pharmacy search engine to work.
- NEVER suggest prescription drugs.
- Do NOT ask about scheduling appointments — another agent handles that.
- Append exactly one line at the end: "I am an AI, not a licensed doctor."

TREATMENT PATH DECISION — choose the most appropriate path:
- "otc" : Use ONLY when the condition is a minor, self-limiting illness (e.g., mild common cold, minor headache, superficial scrape, etc) that clearly does not require a formal medical evaluation. Provide actionable OTC advice + self-care.

- "doctor_needed" : Use whenever symptoms warrant a professional medical evaluation, formal diagnosis, potential prescription therapy, OR if there are moderate to severe concerns (e.g., persistent symptoms, suspected infections like strep or UTI that typically require antibiotics, worsening conditions, undetermined rashes). 

- "emergency" : Immediate life threat only (chest pain, stroke signs, anaphylaxis, unconsciousness, severe bleeding). Direct patient to call 101 immediately.

- "more_info_needed" : Use this if the symptoms are too vague to differentiate between conditions that require different treatment paths (e.g., distinguishing a simple cold from Strep throat or COVID). Aim for the "sweet spot": ask follow-up questions only until you have a solid "clinical feeling" of the diagnosis. Avoid asking more than 3 follow-up questions in total across the conversation.

PROMPT HEURISTIC: Review the conversation history. If you have already asked 2-3 questions and the user has provided enough detail to form a high-probability diagnosis, you MUST proceed to "otc" or "doctor_needed". Do not get stuck in an infinite questioning loop.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "diagnosis": "Probable diagnosis in 1 sentence, or 'Uncertain' if more info needed",
  "action": "otc | doctor_needed | emergency | more_info_needed",
  "reply": "Short, direct message. If more_info_needed, ask your follow-up question. Else, include: (1) diagnosis, (2) what to do including medication name and dosage. Last line: I am an AI, not a licensed doctor.",
  "otc_medications": ["List a MAXIMUM of 1 or 2 essential OTC medication names ONLY in HEBREW (e.g., 'אקמול', 'נורופן'). This is critical for our local pharmacy search engine."],
  "specialist_type": "e.g. General Practitioner — only if doctor_needed, else null",
  "urgency": "immediately | within_24h | this_week | whenever — only if doctor_needed, else null"
}

AGENT MEMORY:
Your memory is provided in the 'Patient Data' block. It contains all symptoms and history extracted by previous agents and your own follow-up turns. You MUST use this data to avoid asking questions about things the patient has already confirmed or denied.
"""


def advise(client: OpenAI, triage: dict, chat_history: list) -> dict:
    """
    Run the Medical Advisor agent.

    Args:
        client : OpenAI-compatible client.
        triage : Triage dict from Agent 1 (symptoms, duration, history).
        chat_history : The full conversation history.
    """
    symptoms = triage.get('symptoms', [])
    medical_context = retrieve_medical_context(symptoms)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # Add history for context (last 5 messages to avoid token bloat)
    for msg in chat_history[-5:]:
        messages.append(msg)

    # Add the current retrieved medical context and patient data
    messages.append({
        "role": "user",
        "content": (
            f"Verified Medical Reference Data:\n{medical_context}\n\n"
            f"Patient Data (Latest Triage):\n"
            f"Symptoms: {', '.join(symptoms)}\n"
            f"Duration: {triage.get('duration', 'unknown')}\n"
            f"Relevant history: {triage.get('relevant_history', 'none')}"
        )
    })

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-OpenRouter-Title": "AhYakar - Medical Advisor",
        },
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "diagnosis": "Unable to determine diagnosis.",
            "action": "doctor_needed",
            "reply": (
                "I wasn't able to fully analyse your situation. "
                "To be safe, I recommend seeing a General Practitioner. "
                "Would you like me to find and book a nearby appointment for you?"
            ),
            "otc_medications": [],
            "specialist_type": "General Practitioner",
            "urgency": "this_week",
        }
