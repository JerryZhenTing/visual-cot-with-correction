"""
Robust parsing utilities for Qwen2.5-VL JSON output.

All public functions accept either:
  - a raw string (model output)
  - an already-parsed dict

Primary entry point: parse_full_output(raw_text, mode)

Individual helpers:
  extract_json_object  – extract first valid JSON object from free text
  parse_reasoning      – get "reasoning" field
  parse_answer         – get "answer" field, normalized to "true"/"false"
  parse_box            – get "box" field, validated for mode
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object from a string.

    Tolerates:
      - Clean JSON output (ideal case)
      - Markdown fences  (```json ... ```)
      - Leading / trailing whitespace or stray text
      - Single-quoted strings (best-effort fix)

    Returns a dict on success, None on failure.
    """
    if not text:
        return None

    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Attempt 1: direct parse (handles the clean case)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the first {...} block via brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        # Try a light repair (single → double quotes, trailing commas)
                        repaired = _repair_json(candidate)
                        try:
                            obj = json.loads(repaired)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break  # only try first candidate

    return None


def _repair_json(text: str) -> str:
    """Light-touch JSON repair for common Qwen output issues."""
    # Replace single-quoted string delimiters with double quotes
    # (only handles simple cases — not nested quotes inside values)
    text = re.sub(r"'([^']*)'", r'"\1"', text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def parse_field(data: Any, field: str) -> Optional[str]:
    """
    Extract any string field by name from parsed JSON or raw text.

    Args:
        data  : dict or raw string
        field : JSON key to extract (e.g. "reasoning", "verification")
    Returns:
        string value or None
    """
    obj = _ensure_dict(data)
    if obj is None:
        return None
    val = obj.get(field)
    return str(val).strip() if val is not None else None


def parse_reasoning(data: Any, key: str = "reasoning") -> Optional[str]:
    """
    Extract a reasoning string by key name.

    Args:
        data: dict or raw string
        key:  JSON key to read — "reasoning" (pass 1) or
              "verification_reasoning" (pass 2 verification)
    Returns:
        string value, or None if not found
    """
    return parse_field(data, key)


def parse_answer(data: Any) -> Optional[str]:
    """
    Extract and normalize the "answer" field.

    Accepts "true" / "false" case-insensitively, or Python bool.
    Returns the lowercase string "true" or "false", or None if invalid.

    Args:
        data: dict or raw string
    """
    obj = _ensure_dict(data)
    if obj is None:
        return None

    raw = obj.get("answer")
    if raw is None:
        return None

    # Python bool
    if raw is True:
        return "true"
    if raw is False:
        return "false"

    normalized = str(raw).strip().lower()
    if normalized in ("true", "false"):
        return normalized

    return None  # unrecognized value


def parse_box(data: Any, mode: str = "visual_cot") -> dict:
    """
    Extract and validate the "box" field.

    Args:
        data: dict or raw string
        mode: "textual_cot" (expects null) or "visual_cot" (expects [x,y,x,y])

    Returns:
        {
          "box": [xmin,ymin,xmax,ymax] or None,
          "valid": bool,
          "invalid_reason": str or None
        }
    """
    obj = _ensure_dict(data)
    box_raw = obj.get("box") if obj else None

    # ---- Textual CoT: box must be null ----
    if mode == "textual_cot":
        if box_raw is None:
            return {"box": None, "valid": True, "invalid_reason": None}
        return {
            "box": None,
            "valid": False,
            "invalid_reason": f"Expected null box for textual_cot, got: {box_raw!r}",
        }

    # ---- Visual CoT: box must be [xmin,ymin,xmax,ymax] in [0,1] ----
    if box_raw is None:
        return {
            "box": None,
            "valid": False,
            "invalid_reason": "box field is null (expected [xmin,ymin,xmax,ymax])",
        }

    if not isinstance(box_raw, (list, tuple)):
        return {
            "box": None,
            "valid": False,
            "invalid_reason": f"box is not a list (got {type(box_raw).__name__})",
        }

    if len(box_raw) != 4:
        return {
            "box": None,
            "valid": False,
            "invalid_reason": f"box has {len(box_raw)} elements (expected 4)",
        }

    try:
        box = [float(v) for v in box_raw]
    except (TypeError, ValueError) as exc:
        return {
            "box": None,
            "valid": False,
            "invalid_reason": f"non-numeric box values: {exc}",
        }

    # Qwen2.5-VL outputs coordinates in [0, 1000] rather than [0, 1].
    # Auto-normalize by dividing by 1000 when any value exceeds 1.
    if any(v > 1.0 for v in box):
        box = [v / 1000.0 for v in box]

    xmin, ymin, xmax, ymax = box

    if not all(0.0 <= v <= 1.0 for v in box):
        return {
            "box": box,
            "valid": False,
            "invalid_reason": f"box values outside [0,1] after normalization: {box}",
        }

    if xmin >= xmax:
        return {
            "box": box,
            "valid": False,
            "invalid_reason": f"xmin ({xmin:.4f}) >= xmax ({xmax:.4f})",
        }

    if ymin >= ymax:
        return {
            "box": box,
            "valid": False,
            "invalid_reason": f"ymin ({ymin:.4f}) >= ymax ({ymax:.4f})",
        }

    return {"box": box, "valid": True, "invalid_reason": None}


# ---------------------------------------------------------------------------
# Composite parser
# ---------------------------------------------------------------------------

def parse_full_output(raw_text: str, mode: str = "textual_cot") -> dict:
    """
    Parse raw model output into all structured fields at once.

    Args:
        raw_text : raw string from model.generate_response()
        mode     : "textual_cot" or "visual_cot"

    Returns:
        {
          "parsed_json"        : dict or None,
          "parsed_reasoning"   : str or None,
          "parsed_answer"      : "true"/"false"/None,
          "parsed_box"         : [xmin,ymin,xmax,ymax] or None,
          "box_valid"          : bool,
          "box_invalid_reason" : str or None,
        }
    """
    parsed_json = extract_json_object(raw_text)
    source = parsed_json if parsed_json is not None else raw_text

    box_result = parse_box(source, mode=mode)

    return {
        "parsed_json": parsed_json,
        "parsed_reasoning": parse_reasoning(source),
        "parsed_answer": parse_answer(source),
        "parsed_box": box_result["box"],
        "box_valid": box_result["valid"],
        "box_invalid_reason": box_result["invalid_reason"],
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ensure_dict(data: Any) -> Optional[dict]:
    """Return a dict from data (dict) or by parsing data (str), or None."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return extract_json_object(data)
    return None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- textual CoT sample ---
    sample_text = '{"reasoning": "The cat is clearly on the left side.", "box": null, "answer": "true"}'
    result = parse_full_output(sample_text, mode="textual_cot")
    print("Textual CoT parse:", result)
    assert result["parsed_answer"] == "true"
    assert result["box_valid"] is True

    # --- visual CoT sample ---
    sample_visual = '{"reasoning": "The dog is above the fence.", "box": [0.1, 0.2, 0.5, 0.8], "answer": "false"}'
    result2 = parse_full_output(sample_visual, mode="visual_cot")
    print("Visual CoT parse:", result2)
    assert result2["parsed_answer"] == "false"
    assert result2["box_valid"] is True
    assert result2["parsed_box"] == [0.1, 0.2, 0.5, 0.8]

    # --- fenced markdown ---
    fenced = '```json\n{"reasoning": "ok", "box": null, "answer": "true"}\n```'
    result3 = parse_full_output(fenced, mode="textual_cot")
    print("Fenced parse:", result3)
    assert result3["parsed_answer"] == "true"

    print("All smoke tests passed.")
