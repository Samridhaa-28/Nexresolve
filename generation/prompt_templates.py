"""
NexResolve Prompt Templates
Contains functions to build prompts for different LLM architectures.
"""
import re


def build_flan_prompt(ticket_summary, intent, entities, retrieved_solution):
    """
    Builds a paragraph-rewrite prompt for FLAN-T5.
    Pre-cleans the solution and asks FLAN to rewrite it professionally.
    """
    # Strip any leftover source labels or tags
    clean = re.sub(r'\[Source \d+\]\n?', '', retrieved_solution).strip()

    return (
        f"Rewrite the following technical solution as a clear, professional paragraph. "
        f"Remove any informal language. Only include actionable information:\n{clean}"
    )


def build_bart_prompt(ticket_summary, retrieved_solution):
    """
    Builds input for BART — pass only the cleaned solution text.
    """
    clean = re.sub(r'\[Source \d+\]\n?', '', retrieved_solution).strip()
    return clean


'''def build_mistral_prompt(ticket_summary, intent, entities, retrieved_solution):
    entities_str = ", ".join([f"{k}: {v}" for k, v in entities.items()]) if isinstance(entities, dict) else str(entities)
    return f"""<s>[INST] System:
You are a helpful IT support assistant.
User:
Ticket:
{ticket_summary}
Intent:
{intent}
Entities:
{entities_str}
Relevant Solution:
{retrieved_solution}
[/INST] Assistant:
Response:"""'''