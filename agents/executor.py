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
    Execute the appointment booking.

    In simulation mode (BOOKING_API_URL not set): logs the request and returns
    a simulated confirmation.

    In live mode: POSTs to BOOKING_API_URL and returns the real confirmation.

    Args:
        appointment_details : Dict from the Appointment Planner agent.
        patient_context     : Triage dict to attach as context for the API.

    Returns:
        Dict with keys: success (bool), simulated (bool), confirmation_id, message.

    Raises:
        RuntimeError: If the live API call fails.
    """
    payload = {
        "appointment_type": appointment_details.get("type"),
        "specialist_type": appointment_details.get("specialist_type"),
        "urgency": appointment_details.get("urgency"),
        "reason": appointment_details.get("reason"),
        "patient_context": patient_context or {},
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Simulation mode ─────────────────────────────────────────────────────
    if not BOOKING_API_URL:
        sim_id = f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info("[Executor] SIMULATION – booking payload: %s", payload)
        urgency_label = (appointment_details.get("urgency") or "this_week").replace("_", " ")
        return {
            "success": True,
            "simulated": True,
            "confirmation_id": sim_id,
            "message": (
                f"✅ Appointment request logged!\n"
                f"Doctor: {appointment_details.get('specialist_type', 'Doctor')}\n"
                f"Urgency: {urgency_label}\n"
                f"Reference: {sim_id}\n\n"
                f"_(Simulation mode — set BOOKING_API_URL to connect to a real booking system.)_"
            ),
        }

    # ── Live mode ────────────────────────────────────────────────────────────
    try:
        import requests  # imported lazily so it's only required in live mode

        resp = requests.post(BOOKING_API_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        confirmation_id = data.get("id") or data.get("confirmation_id") or "N/A"
        details = data.get("details", "")

        logger.info("[Executor] Booking confirmed: %s", confirmation_id)
        return {
            "success": True,
            "simulated": False,
            "confirmation_id": confirmation_id,
            "message": (
                f"✅ Appointment booked successfully!\n"
                f"Doctor: {appointment_details.get('specialist_type', 'Doctor')}\n"
                f"Confirmation ID: {confirmation_id}\n"
                + (f"Details: {details}" if details else "")
            ),
        }

    except Exception as exc:
        logger.error("[Executor] Booking API call failed: %s", exc)
        raise RuntimeError(f"Booking API error: {exc}") from exc
