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
    specialist_type = appointment_details.get("specialist_type", "Family Doctor")
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
            # Added flags to fix the "transparent window" issue in WSL/Linux.
            browser = p.chromium.launch(
                headless=False, 
                slow_mo=50,
                args=[
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-software-rasterizer'
                ]
            )
            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            # 1. Navigate to Mock Maccabi Online
            logger.info("[Executor] Navigating to Mock Maccabi online (localhost:7800)...")
            page.goto("http://127.0.0.1:7800/", timeout=30000)

            # 1.5 Wait for user to manually login
            logger.info("[Executor] Waiting for user to log in manually (up to 60 seconds)...")
            try:
                # Wait until the 'doctors' nav item is visible, which implies login success
                page.wait_for_selector('div[data-page="doctors"]', state='visible', timeout=60000)
                logger.info("[Executor] Login passed. Continuing flow...")
            except PlaywrightTimeoutError:
                raise RuntimeError("Login timed out or failed. Did not reach the dashboard.")

            # 2. Go to Doctors page
            logger.info("[Executor] Navigating to 'Find Doctors' page...")
            page.click('div[data-page="doctors"]', timeout=5000)
            page.wait_for_timeout(1000)

            # 3. Select the correct specialist filter pill
            logger.info(f"[Executor] Looking for specialty filter matching: {specialist_type}")
            page.wait_for_selector('.filter-chip', timeout=5000)
            
            chips = page.locator('.filter-chip')
            chip_count = chips.count()
            
            best_match = None
            target_text = specialist_type.lower()
            target_first_word = target_text.split()[0] if target_text else ""
            
            for i in range(chip_count):
                chip_text = chips.nth(i).inner_text().strip().lower()
                
                if chip_text == target_text:
                    best_match = chips.nth(i)
                    break
                elif target_first_word and target_first_word in chip_text:
                    # e.g., "family doctor" matches "family medicine"
                    best_match = chips.nth(i)
            
            if best_match:
                best_match.click()
                logger.info(f"[Executor] Clicked filter chip for: {specialist_type}")
            else:
                logger.warning(f"[Executor] Could not find a filter chip matching {specialist_type}. Falling back to search bar.")
                # Fallback to search bar if pill isn't found
                search_input_selector = '#doctor-search'
                page.fill(search_input_selector, specialist_type)
                page.press(search_input_selector, "Enter")

            # Wait for search results
            page.wait_for_timeout(1500)

            # 4. Order to the best available one
            logger.info("[Executor] Attempting to select the best available appointment...")
            first_book_btn = page.locator('#doctors-grid button, .doctor-card button').first
            first_book_btn.wait_for(state="visible", timeout=5000)
            first_book_btn.click()
            logger.info("[Executor] Clicked book on the first available doctor.")

            # 5. Handle booking modal
            logger.info("[Executor] Simulating booking confirmation in modal...")
            page.wait_for_selector('#booking-modal', timeout=5000)
            page.wait_for_timeout(500) # Give UI a moment to populate modal
            
            # Extract actual doctor details from the modal
            doctor_name = page.locator('#modal-doctor-name').inner_text().strip() or specialist_type
            clinic_location = page.locator('#modal-doctor-clinic').inner_text().strip() or "Local Clinic"
            
            target_date = "2026-06-01"
            page.fill('#booking-date', target_date)
            page.wait_for_timeout(500)
            
            time_slots = page.locator('.time-slot')
            booked_time = "TBD"
            if time_slots.count() > 0:
                booked_time = time_slots.first.inner_text().strip()
                time_slots.first.click()
                page.wait_for_timeout(500)

            page.click('#confirm-booking')
            logger.info("[Executor] Clicked Confirm Booking!")
            
            # Let's pause so the user can see the browser open during the hackathon demo
            page.wait_for_timeout(3000) 

            # Generate a mock confirmation ID since we are simulating the final step
            confirmation_id = f"MAC-{datetime.now().strftime('%Y%m%d%H%M')}"
            
            browser.close()

            return {
                "success": True,
                "simulated": False,  # True web scraping was used on the mock site
                "confirmation_id": confirmation_id,
                "message": (
                    f"✅ Appointment booked successfully!\n\n"
                    f"**Doctor:** {doctor_name}\n"
                    f"**Location:** {clinic_location}\n"
                    f"**Date & Time:** {target_date} at {booked_time}\n"
                    f"**Confirmation ID:** {confirmation_id}\n\n"
                    f"_(Note: Executed against the mock KupatHolim site at localhost:7800)_"
                ),
            }

    except Exception as exc:
        logger.error("[Executor] Web scraping failed: %s", exc)
        raise RuntimeError(f"Web scraping error: {exc}") from exc
