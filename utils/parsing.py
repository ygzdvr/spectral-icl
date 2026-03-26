"""Parsing helpers reused by experiment entry-point scripts.

These helpers centralize the common "comma-separated CLI string -> typed list"
pattern used throughout the `scripts/` directory. Keeping parsing in one place:

- avoids repeating split/strip/filter logic in every script,
- guarantees consistent handling of trailing commas and spaces, and
- makes future validation changes easy to apply repo-wide.
"""

from __future__ import annotations


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated string into integers.

    Expected examples:
        - ``"1,2,4,8"`` -> ``[1, 2, 4, 8]``
        - ``" 1, 2 , 4 ,"`` -> ``[1, 2, 4]``

    Behavior notes:
        - Surrounding whitespace is removed per token.
        - Empty tokens are ignored (for robust CLI ergonomics).
        - Any non-integer token raises ``ValueError`` from ``int(...)``.

    Args:
        raw: Raw command-line string value (usually from ``argparse``).

    Returns:
        Parsed integer list in original token order.
    """
    # Split once on commas and normalize whitespace token-by-token.
    # The conditional filters out empty tokens caused by ",," or trailing commas.
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated string into floats.

    Expected examples:
        - ``"0.1,0.2,0.4"`` -> ``[0.1, 0.2, 0.4]``
        - ``" 1, 2.5 , 3e-1 ,"`` -> ``[1.0, 2.5, 0.3]``

    Behavior notes:
        - Surrounding whitespace is removed per token.
        - Empty tokens are ignored (for robust CLI ergonomics).
        - Any non-float token raises ``ValueError`` from ``float(...)``.

    Args:
        raw: Raw command-line string value (usually from ``argparse``).

    Returns:
        Parsed float list in original token order.
    """
    # Same tokenization policy as `parse_int_list`, but converted with float(...)
    # to support decimals and scientific notation.
    return [float(part.strip()) for part in raw.split(",") if part.strip()]
