def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens from a string.

    Based on the rough heuristic that 1 token is ~1.7 words.
    """
    if not text:
        return 0
    words = text.split()
    return round(len(words) / 1.7)
