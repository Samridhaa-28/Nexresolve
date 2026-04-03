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
        # Split after the last "Response:" or "Assistant:" marker
        # We look for "Response:" first, then fallback to "Assistant:"
        if "Response:" in text:
            text = text.split("Response:")[-1].strip()
        elif "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
        # Remove any lingering chat markers
        text = text.replace("</s>", "").strip()'''
        
    return text

def enforce_length(text, soft_cap=150, hard_cap=200):
    """
    Enforces word length constraints. 
    Cuts at the last sentence boundary before the hard cap.
    """
    words = text.split()
    
    # If within soft cap, return as is
    if len(words) <= soft_cap:
        return text
    
    # If over soft cap, we need to consider trimming
    # We aim for something between soft_cap and hard_cap
    truncated_words = words[:hard_cap]
    truncated_text = " ".join(truncated_words)
    
    # Find last sentence boundary (., !, ?)
    last_boundary = max(
        truncated_text.rfind("."), 
        truncated_text.rfind("!"), 
        truncated_text.rfind("?")
    )
    
    if last_boundary != -1:
        return truncated_text[:last_boundary + 1].strip()
    
    # Fallback if no sentence boundary found, just return hard cap (though not ideal)
    return truncated_text.strip()

def normalize_tone(text):
    """
    Ensures a polite, clear, and actionable tone.
    Removes abrupt endings and ensures complete sentences.
    """
    text = text.strip()
    
    if not text:
        return "I'm sorry, I couldn't generate a solution at this time. Please try again or contact support directly."

    # Ensure it ends with a punctuation
    if text[-1] not in [".", "!", "?"]:
        # If it doesn't end with punctuation, it might be cut off
        # We try to find the last complete sentence
        last_boundary = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_boundary != -1:
            text = text[:last_boundary + 1]
        else:
            text += "."
            
    # Add a polite closing if not present
    polite_closings = ["hope this helps", "let me know if you need more assistance", "have a great day"]
    if not any(closing in text.lower() for closing in polite_closings):
        # We don't necessarily want to force it every time, but ensure it's not abrupt
        pass
        
    return text

'''def enforce_step_structure(text, retrieved_solution):
    """
    Ensures output follows original steps exactly.
    """
    steps = re.split(r'\d+\.\s*', retrieved_solution)
    
    # Remove empty entries
    steps = [s.strip() for s in steps if s.strip()]

    # Rebuild properly numbered steps
    cleaned = []
    for i, step in enumerate(steps, 1):
        cleaned.append(f"{i}. {step}")

    return "\n".join(cleaned)'''
