"""Reading a currency code out of a preformatted price.

MIRROR of the chatbot's `helpers/money.py`, cut down to the one function the
indexer needs. The symbol table has to agree with the chatbot's, because a
product indexed here and read there must report the same currency.
"""
from __future__ import annotations

from typing import Optional

SYMBOLS = {
    "ILS": "₪",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "TRY": "₺",
    "SAR": "﷼",
    "AED": "د.إ",
    "JOD": "د.ا",
    "EGP": "£",
}

# Longest symbols first, so "د.إ" is not read as "د.ا".
_BY_SYMBOL = sorted(((s, c) for c, s in SYMBOLS.items()),
                    key=lambda pair: -len(pair[0]))


def currency_from_text(text: Optional[str]) -> Optional[str]:
    """Currency code from a price like "₪200.00", or None."""
    if not text:
        return None
    for symbol, code in _BY_SYMBOL:
        if symbol in str(text):
            return code
    return None
