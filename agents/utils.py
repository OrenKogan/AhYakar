"""Shared utilities for the AhYakar agent pipeline."""

import json
import re


def parse_json_response(text: str) -> dict:
    """
    Robustly parse a JSON object from an LLM response.
    Handles markdown code fences (```json ... ```) that models sometimes add.
    """
    text = text.strip()
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    return json.loads(text)
