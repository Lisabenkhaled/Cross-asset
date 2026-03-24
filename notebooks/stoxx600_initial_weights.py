"""STOXX 600 sector ETF baseline weights used as neutral allocation reference."""

from __future__ import annotations

STOXX600_INITIAL_WEIGHTS = {
    "SXNP": 16.453337,
    "SX7P": 14.827153,
    "SXDP": 14.417925,
    "SXIP": 5.922743,
    "S600ENP": 5.808279,
    "SX8P": 5.521192,
    "S600FOP": 5.303101,
    "S600CPP": 4.762060,
    "SX6P": 4.758787,
    "SXFP": 4.065497,
    "SXOP": 3.994310,
    "SXKP": 3.076568,
    "SXPP": 2.665627,
    "S600PDP": 2.167012,
    "SX4P": 1.581998,
    "SXAP": 1.470205,
    "SX86P": 1.310971,
    "SXTP": 0.838710,
    "SXRP": 0.641078,
    "SXMP": 0.413447,
}

# ensure sum = 100%
def normalized_initial_weights() -> dict[str, float]:
    """Return weights normalized to 100%."""
    total = sum(STOXX600_INITIAL_WEIGHTS.values())
    return {k: (v / total) * 100 for k, v in STOXX600_INITIAL_WEIGHTS.items()}
