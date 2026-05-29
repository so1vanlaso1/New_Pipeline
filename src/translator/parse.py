"""Extract the <z3py> block from translator output, with fallbacks for stray fences."""

from __future__ import annotations

import re

_Z3PY_TAG = re.compile(r"<z3py>\s*(.*?)\s*</z3py>", re.DOTALL | re.IGNORECASE)
_FENCE_PYTHON = re.compile(r"```(?:python|py|z3py)?\s*(.*?)```", re.DOTALL)


def parse_translator_output(text: str) -> str | None:
    """Return the Z3-Python program string, or None on parse failure."""
    m = _Z3PY_TAG.search(text)
    if m:
        return m.group(1).strip() or None
    fences = _FENCE_PYTHON.findall(text)
    if fences:
        return fences[0].strip() or None
    # Last resort: if the text already looks like Python (contains `premises =` and `goal =`),
    # use it as-is.
    if "premises" in text and "goal" in text and "DeclareSort" in text:
        return text.strip()
    return None
