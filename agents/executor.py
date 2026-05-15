"""
Agent 4 — Executor
==================
Books the appointment by calling an external booking API.

Configuration:
    Set the BOOKING_API_URL environment variable to the real endpoint when
    the API is available. Until then, the agent runs in simulation mode and
    logs the request without actually calling any external service.

Expected API contract (POST to BOOKING_API_URL):
    Request body (JSON):
    {
        "appointment_type"  : "gp" | "specialist",
        "specialist_type"   : "Family Doctor" | "Cardiologist" | ...,
        "urgency"           : "immediately" | "within_24h" | "this_week" | "whenever",
        "reason"            : "brief medical reason",
        "patient_context"   : { ...triage fields... },
        "requested_at"      : "ISO-8601 UTC timestamp"
    }

    Expected success response (JSON) — at least one of:
    {
        "id"              : "booking reference ID",
        "confirmation_id" : "booking reference ID",
        "details"         : "human-readable booking details (optional)"
    }
"""

import asyncio
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.openai.chat import ChatOpenAI  # Use browser-use's own OpenAI wrapper (compatible ainvoke)
import logging
from datetime import datetime, timezone
import os


logger = logging.getLogger(__name__)

async def run_browser_agent(specialist_type: str, urgency: str) -> str:
    """Runs the asynchronous browser-use agent."""
    # Initialize the LLM via OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model="openai/gpt-4o"  # Defaulting to gpt-4o for best visual reasoning
    )

    task_prompt = (
        f"You are an autonomous medical booking agent. Your strict task is to book a '{specialist_type}' appointment.\n"
        f"1. Go to http://127.0.0.1:7999/.\n"
        f"2. You MUST log in. Type 'a@gmail.com' into the email input, type '123123' into the password input, and click the Login button.\n"
        f"3. Wait for the dashboard to load, then click on the 'Find Doctors' navigation tab.\n"
        f"4. Find the specialty filters and click the pill that best matches '{specialist_type}'.\n"
        f"5. Click the 'Book' button for the first doctor in the results list.\n"
        f"6. In the booking modal, select tomorrow's date. If unable to find a time slot for tomorrow, pick the next available day and select a time slot.\n"
        f"7. Click the 'Confirm Booking' button.\n"
        f"8. DO NOT complete the task until you actually see the confirmation screen. Once confirmed, extract the doctor's name, the clinic location, and the booked time slot, and return them."
    )

    # WSL display flags + speed tuning: minimal waits between actions
    profile = BrowserProfile(
        args=[
            '--disable-gpu',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-software-rasterizer'
        ],
        minimum_wait_page_load_time=0.3,          # Was ~1s default — cut to 300ms
        wait_for_network_idle_page_load_time=1.0,  # Was ~3s default — cut to 1s
        wait_between_actions=0.3,                  # Was ~1s default — cut to 300ms
    )

    browser_session = BrowserSession(browser_profile=profile)

    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser=browser_session,
        flash_mode=True,           # Strips planning overhead — much faster action loop
        max_actions_per_step=5,    # Chain more clicks per LLM call
        use_vision=True,
    )

    try:
        result = await agent.run()
        final_output = result.final_result() if result else None
        if not final_output:
            raise RuntimeError("Agent completed without a confirmed result. Booking may not have succeeded.")
        return final_output
    finally:
        # Always close the browser window when done (success or failure)
        await browser_session.stop()

def execute_booking(appointment_details: dict, patient_context: dict | None = None) -> dict:
    """
    Execute the appointment booking using an autonomous visual web agent.

    Args:
        appointment_details : Dict from the Appointment Planner agent.
        patient_context     : Triage dict to attach as context.

    Returns:
        Dict with keys: success (bool), simulated (bool), confirmation_id, message.
    """
    specialist_type = appointment_details.get("specialist_type", "Family Doctor")
    urgency = appointment_details.get("urgency", "this_week")
    
    logger.info("[Executor] Starting visual web agent to book appointment for: %s", specialist_type)

    try:
        # Run the async agent synchronously
        final_output = asyncio.run(run_browser_agent(specialist_type, urgency))
        
        # Only report success if the agent returned a real result
        confirmation_id = f"MAC-{datetime.now().strftime('%Y%m%d%H%M')}"
        
        return {
            "success": True,
            "simulated": False,
            "confirmation_id": confirmation_id,
            "message": (
                f"✅ Appointment booked successfully via AI Visual Agent!\n\n"
                f"**Details:**\n{final_output}\n\n"
                f"**Confirmation ID:** {confirmation_id}\n\n"
                f"_(Executed autonomously against the KupatHolim portal)_"
            ),
        }

    except Exception as exc:
        logger.error("[Executor] Visual web agent failed: %s", exc)
        raise RuntimeError(f"Web agent error: {exc}") from exc
