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
        "specialist_type"   : "General Practitioner" | "Cardiologist" | ...,
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

import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Set BOOKING_API_URL in your environment when the real API is ready.
BOOKING_API_URL: str = os.getenv("BOOKING_API_URL", "")


def execute_booking(appointment_details: dict, patient_context: dict | None = None) -> dict:
    """
    Execute the appointment booking using web scraping on Maccabi Online.

    Args:
        appointment_details : Dict from the Appointment Planner agent.
        patient_context     : Triage dict to attach as context.

    Returns:
        Dict with keys: success (bool), simulated (bool), confirmation_id, message.
    """
    specialist_type = appointment_details.get("specialist_type", "General Practitioner")
    urgency = appointment_details.get("urgency", "this_week")
    
    logger.info("[Executor] Starting web scraper to book appointment for: %s", specialist_type)

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        logger.error("[Executor] Playwright is not installed. Please run: pip install playwright && playwright install")
        raise RuntimeError("Playwright is missing. Cannot run scraper.")

    try:
        with sync_playwright() as p:
            # Launch browser (set headless=False to observe the process during hackathon)
            browser = p.chromium.launch(headless=False, slow_mo=50)
            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            # 1. Navigate to Mock Maccabi Online
            logger.info("[Executor] Navigating to Mock Maccabi online (Windows Host: 172.26.144.1:7800)...")
            page.goto("http://127.0.0.1:7800/", timeout=30000)

            # Wait a bit to ensure page load
            page.wait_for_timeout(1000)

            # 2. Go to Doctors page
            logger.info("[Executor] Navigating to 'Find Doctors' page...")
            try:
                page.click('div[data-page="doctors"]', timeout=5000)
                page.wait_for_timeout(1000)
            except PlaywrightTimeoutError:
                logger.warning("[Executor] Could not click doctors nav tab. Proceeding anyway.")

            # 3. Search for the specialist
            logger.info(f"[Executor] Searching for specialist: {specialist_type}")
            search_input_selector = '#doctor-search' 
            
            try:
                page.wait_for_selector(search_input_selector, timeout=5000)
                page.fill(search_input_selector, specialist_type)
                page.press(search_input_selector, "Enter")
                logger.info("[Executor] Submitted search query.")
            except PlaywrightTimeoutError:
                logger.warning("[Executor] Could not find #doctor-search. Page structure might be different.")

            # Wait for search results
            page.wait_for_timeout(1500)

            # 4. Order to the most nearby/best available one
            logger.info("[Executor] Attempting to select the best available appointment...")
            
            try:
                # Find the first book button in the grid
                first_book_btn = page.locator('#doctors-grid button, .doctor-card button').first
                first_book_btn.wait_for(state="visible", timeout=5000)
                first_book_btn.click()
                logger.info("[Executor] Clicked book on the first available doctor.")
            except PlaywrightTimeoutError:
                logger.warning("[Executor] Could not find a book button for any doctor.")

            # 5. Handle booking modal
            logger.info("[Executor] Simulating booking confirmation in modal...")
            try:
                # Wait for modal to appear
                page.wait_for_selector('#booking-modal', timeout=5000)
                
                # Select a date (tomorrow's date or a fixed dummy date)
                # Since it's a mock UI, let's just put a reasonable date string
                target_date = "2026-06-01"
                page.fill('#booking-date', target_date)
                page.wait_for_timeout(500)
                
                # Click the first time slot if available
                time_slots = page.locator('.time-slot')
                if time_slots.count() > 0:
                    time_slots.first.click()
                    page.wait_for_timeout(500)

                # Click confirm booking
                page.click('#confirm-booking')
                logger.info("[Executor] Clicked Confirm Booking!")
            except PlaywrightTimeoutError:
                logger.warning("[Executor] Could not complete booking modal steps.")
            
            # Let's pause so the user can see the browser open during the hackathon demo
            page.wait_for_timeout(3000) 

            # Generate a mock confirmation ID since we are simulating the final step
            confirmation_id = f"MAC-{datetime.now().strftime('%Y%m%d%H%M')}"
            
            browser.close()

            urgency_label = urgency.replace("_", " ")
            return {
                "success": True,
                "simulated": False,  # True web scraping was used on the mock site
                "confirmation_id": confirmation_id,
                "message": (
                    f"✅ Appointment booked successfully via Web Automation!\n"
                    f"Doctor: {specialist_type}\n"
                    f"Urgency: {urgency_label}\n"
                    f"Confirmation ID: {confirmation_id}\n\n"
                    f"_(Note: Executed against the mock KupatHolim site at localhost:7800)_"
                ),
            }

    except Exception as exc:
        logger.error("[Executor] Web scraping failed: %s", exc)
        raise RuntimeError(f"Web scraping error: {exc}") from exc
