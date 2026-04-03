"""
NexResolve Prompt Templates
Contains functions to build prompts for different LLM architectures.
"""

def build_flan_prompt(ticket_summary, intent, entities, retrieved_solution):
    """
    Builds a simplified instruction prompt for FLAN-T5.
    """
    return f"""You are a customer support assistant.

Ticket: {ticket_summary}
Intent: {intent}

Solution:
{retrieved_solution}

Rewrite the solution clearly in simple step-by-step instructions for the user.
"""

def build_bart_prompt(ticket_summary, retrieved_solution):
    """
    Builds input for BART using only the retrieved solution.
    """
    return retrieved_solution

'''def build_mistral_prompt(ticket_summary, intent, entities, retrieved_solution):
    """
    Builds a chat-based prompt for Mistral.
    """
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
