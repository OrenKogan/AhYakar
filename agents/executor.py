"""
Agent 4 — Executor
==================
Two-phase autonomous booking agent:

  Phase 1 — scan_appointments():
    Opens the portal, logs in, filters by specialty, and collects 3 appointment
    options from different doctors. Closes the browser cleanly when done.

  Phase 2 — execute_choice():
    Opens a FRESH browser session (the previous asyncio event loop is gone),
    logs in, navigates to the chosen doctor, and confirms the booking.

Note on session sharing: browser_use ties its event bus to the asyncio event
loop. When asyncio.run() returns, the loop closes and the session becomes dead.
Running each phase with its own asyncio.run() + fresh BrowserSession is the
only reliable pattern when calling from synchronous Flask code.
"""

import asyncio
import json
import re
import time
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.openai.chat import ChatOpenAI
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Portal credentials
PORTAL_URL      = "http://127.0.0.1:7999/"
PORTAL_EMAIL    = "a@gmail.com"
PORTAL_PASSWORD = "abcd23asd12"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model="openai/gpt-4o",
    )

def _make_profile() -> BrowserProfile:
    """Headless, speed-tuned profile with WSL/Linux rendering flags."""
    return BrowserProfile(
        headless=True,   # Works with use_vision=False — no screenshot rendering needed
        args=[
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-software-rasterizer",
        ],
        minimum_wait_page_load_time=0.3,
        wait_for_network_idle_page_load_time=1.0,
        wait_between_actions=0.3,
    )

async def _run_agent(task: str, max_steps: int = 20) -> str:
    """Create a fresh browser session, run the agent, always close when done."""
    llm = _make_llm()
    session = BrowserSession(browser_profile=_make_profile())
    agent = Agent(
        task=task,
        llm=llm,
        browser=session,
        flash_mode=True,
        max_actions_per_step=5,
        max_steps=max_steps,
        use_vision=False,   # DOM-based navigation — no screenshots, faster + headless-compatible
    )
    try:
        result = await agent.run()
        return result.final_result() if result else ""
    finally:
        try:
            await session.stop()
        except Exception:
            pass


# ── Phase 1: Scan appointments ────────────────────────────────────────────────

SCAN_PROMPT_TEMPLATE = """You are a medical appointment scanner.

1. Go to http://127.0.0.1:7999/
2. Log in: email = 'a@gmail.com', password = 'abcd23asd12'. Click Login.
3. Navigate to the 'Find Doctors' tab.
4. Click the specialty filter pill that best matches 'SPECIALTY_PLACEHOLDER'.
5. Collect 3 appointment options from the list of available doctors:
   - Open the first doctor's 'Book' modal and note one available time slot.
   - Close that modal. Open the second doctor's 'Book' modal and note a different time slot.
   - Close that modal. Open the third doctor's 'Book' modal and note another time slot.
   IMPORTANT: Each option must be from a DIFFERENT doctor. Do NOT try to change dates
   or search for dates that are not shown - just use whatever dates are available.
   Do NOT click the 'Confirm Booking' button.
6. Return ONLY a valid JSON array (no markdown, no extra text):
[
  {"doctor": "Dr. Name", "clinic": "Clinic Name", "date": "YYYY-MM-DD", "time": "HH:MM"},
  {"doctor": "Dr. Name", "clinic": "Clinic Name", "date": "YYYY-MM-DD", "time": "HH:MM"},
  {"doctor": "Dr. Name", "clinic": "Clinic Name", "date": "YYYY-MM-DD", "time": "HH:MM"}
]"""


def scan_appointments(appointment_details: dict, flask_session_id: str = "") -> dict:
    """
    Phase 1: Scan the portal and return up to 3 appointment options.
    Opens a fresh browser, closes it when done.
    """
    specialist_type = appointment_details.get("specialist_type", "Family Doctor")
    logger.info("[Executor] Phase 1 — Scanning for: %s", specialist_type)
    try:
        # Use simple string replace (not .format) to avoid conflicts with JSON braces
        prompt = SCAN_PROMPT_TEMPLATE.replace("SPECIALTY_PLACEHOLDER", specialist_type)
        raw = asyncio.run(_run_agent(prompt))
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            raise RuntimeError(f"Agent did not return a JSON list. Got: {raw[:300]}")
        options = json.loads(match.group())
        logger.info("[Executor] Found %d options", len(options))
        return {"success": True, "options": options[:3], "specialist_type": specialist_type}
    except Exception as exc:
        logger.error("[Executor] Scan failed: %s", exc)
        raise RuntimeError(f"Appointment scan error: {exc}") from exc


# ── Phase 2: Book chosen appointment ─────────────────────────────────────────

BOOK_PROMPT_TEMPLATE = """You are a medical appointment booking agent.

Book this SPECIFIC appointment:
  - Doctor:    DOCTOR_PLACEHOLDER
  - Clinic:    CLINIC_PLACEHOLDER
  - Date:      DATE_PLACEHOLDER
  - Time:      TIME_PLACEHOLDER
  - Specialty: SPECIALTY_PLACEHOLDER

Steps:
1. Go to http://127.0.0.1:7999/
2. Log in: email = 'a@gmail.com', password = 'abcd23asd12'. Click Login.
3. Wait for the dashboard to load.
4. Click the 'Find Doctors' navigation tab.
5. Click the specialty filter pill that best matches 'SPECIALTY_PLACEHOLDER'.
6. Find the doctor named 'DOCTOR_PLACEHOLDER' in the results list and click their 'Book' button.
7. In the booking modal, the date 'DATE_PLACEHOLDER' should already be selected.
   Click on the time slot 'TIME_PLACEHOLDER' to select it.
8. Click the 'Confirm Booking' (or 'Confirm Appointment') button.
9. As soon as you click the confirm button, consider the booking successful. DO NOT wait for or look for a confirmation screen.
10. Return: doctor name, clinic, date, and confirmed time."""


def execute_choice(chosen_option: dict, specialist_type: str, flask_session_id: str = "") -> dict:
    """
    Phase 2: Book the specific appointment the user chose.
    Opens a fresh browser, logs in, books, closes when done.
    """
    logger.info("[Executor] Phase 2 — Booking: %s", chosen_option)
    # Wait briefly for Phase 1's event loop watchdogs to fully clean up
    # before we start a new asyncio.run(). This prevents QueueShutDown crashes.
    time.sleep(3)
    try:
        # Use simple string replace (not .format) to avoid conflicts with JSON braces
        prompt = (
            BOOK_PROMPT_TEMPLATE
            .replace("DOCTOR_PLACEHOLDER",    chosen_option.get("doctor", ""))
            .replace("CLINIC_PLACEHOLDER",    chosen_option.get("clinic", ""))
            .replace("DATE_PLACEHOLDER",      chosen_option.get("date", ""))
            .replace("TIME_PLACEHOLDER",      chosen_option.get("time", ""))
            .replace("SPECIALTY_PLACEHOLDER", specialist_type)
        )
        final_output = asyncio.run(_run_agent(prompt))
        if not final_output:
            raise RuntimeError("Agent completed without confirming the booking.")

        raw_date = chosen_option.get("date", "")
        raw_time = chosen_option.get("time", "")
        try:
            from datetime import datetime as _dt
            d = _dt.strptime(raw_date, "%Y-%m-%d")
            display_date = f"{d.day}.{d.month}"
        except Exception:
            display_date = raw_date

        return {
            "success": True,
            "confirmation_id": f"MAC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "message": f"\u2705 Booked: {chosen_option.get('doctor', '')}  {raw_time}  {display_date}  \u2014  {chosen_option.get('clinic', '')}",
        }
    except Exception as exc:
        logger.error("[Executor] Booking failed: %s", exc)
        raise RuntimeError(f"Booking error: {exc}") from exc


# ── Legacy entrypoint (kept for compatibility) ────────────────────────────────

def execute_booking(appointment_details: dict, patient_context: dict | None = None) -> dict:
    """
    Legacy single-shot booking. Scans and books the first available option.
    """
    scan = scan_appointments(appointment_details)
    options = scan.get("options", [])
    if not options:
        raise RuntimeError("No appointment options found on the portal.")
    return execute_choice(options[0], scan["specialist_type"])
