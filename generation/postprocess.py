"""
NexResolve Postprocessing Module
Contains utilities for cleaning and formatting model-generated responses.
"""
import re


def clean_output(text, model_type):
    """
    Cleans output based on model type.
    """
    text = text.strip()

    '''if model_type.upper() == "MISTRAL":
        if "Response:" in text:
            text = text.split("Response:")[-1].strip()
        elif "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
        text = text.replace("</s>", "").strip()'''

    return text


def _safe_sentence_boundary(text, cap):
    """
    Finds the last safe sentence-ending boundary within `cap` characters.
    Skips dots inside technical identifiers like tf.test.foo() or module.method.
    """
    chunk = text[:cap]
    for pos in reversed(range(len(chunk))):
        ch = chunk[pos]
        if ch not in ".!?":
            continue
        before = chunk[pos - 1] if pos > 0 else " "
        after  = chunk[pos + 1] if pos + 1 < len(chunk) else " "
        # Skip if dot is inside an identifier (e.g. tf.test or module.method)
        if before.isalpha() and after.islower():
            continue
        return pos
    return -1


def enforce_length(text, max_words=80):
    """
    Caps paragraph at max_words without cutting inside technical identifiers.
    """
    if not text or not text.strip():
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    boundary = _safe_sentence_boundary(truncated, len(truncated))
    if boundary != -1:
        return truncated[:boundary + 1].strip()
    return truncated.strip()


def normalize_tone(text):
    """
    Light cleanup — removes filler openers FLAN sometimes adds.
    Never truncates the text.
    """
    text = text.strip()

    if not text:
        return (
            "I'm sorry, I couldn't generate a solution at this time. "
            "Please try again or contact support directly."
        )

    # Remove common filler openers
    filler_patterns = [
        r'^(Sure|Certainly|Of course|Absolutely)[,!.]?\s*',
        r'^Here (are|is) (the|your)?.{0,30}(steps?|solution|answer)[.:]\s*',
        r'^The (solution|answer|fix)[.:]\s*',
        r'^To (fix|resolve|solve).{0,60}:\s*',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Ensure ends with punctuation
    text = text.strip()
    if text and text[-1] not in ".!?":
        boundary = _safe_sentence_boundary(text, len(text))
        if boundary != -1:
            text = text[:boundary + 1]
        else:
            text += "."

    return text.strip()


'''def enforce_step_structure(text, retrieved_solution):
    steps = re.split(r"\d+\.\s*", retrieved_solution)
    steps = [s.strip() for s in steps if s.strip()]
    cleaned = []
    for i, step in enumerate(steps, 1):
        cleaned.append(f"{i}. {step}")
    return "\n".join(cleaned)'''