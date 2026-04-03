"""
Verification Script for NexResolve Response Generator
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.generator import ResponseGenerator

'''try:
    from nexresolve.generation.generator import ResponseGenerator
except ImportError:
    # Handle if run from within scripts dir or root
    try:
        from generation.generator import ResponseGenerator
    except ImportError:
        print("Could not import ResponseGenerator. Ensure you are running from the project root.")
        sys.exit(1)'''

def main():
    print("=== NexResolve Generator Test ===")
    
    # Mock data
    ticket_summary = "My laptop is not connecting to the office WiFi after the update."
    intent = "network_issue"
    entities = {"device": "laptop", "issue": "wifi_connection"}
    retrieved_solution = "1. Restart the router. 2. Check for driver updates. 3. Reset network settings."

    print("\nParameters:")
    print(f"- Ticket: {ticket_summary}")
    print(f"- Intent: {intent}")
    print(f"- Solution: {retrieved_solution}")

    # For testing purposes without actual model loading (which might take time/resources)
    # We can use a mocked class if needed, or just attempt loading FLAN-T5 (smallest)
    
    try:
        # We start with FLAN as it's the base model
        models = ["FLAN", "BART"]

        for model in models:
          print(f"\n--- Testing {model} ---")
    
          generator = ResponseGenerator(model_type=model)
    
          print("Generating Response...")
          response = generator.generate(
           ticket_summary, 
           intent, 
           entities, 
           retrieved_solution
          )
    
          print("\n=== FINAL RESPONSE ===")
          print(response)
          print("=======================")
          
          del generator

          import torch
          if torch.cuda.is_available():
             torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[ERROR] Actual model loading failed: {str(e)}")
        print("\nFalling back to logic verification (Mock Output)...")
        # Just to show the logic/prompt would work if models were loaded
        from generation.prompt_templates import build_flan_prompt
        from generation.postprocess import clean_output, enforce_length, normalize_tone
        
        prompt = build_flan_prompt(ticket_summary, intent, entities, retrieved_solution)
        print("\nMock Prompt (FLAN):")
        print("-" * 20)
        print(prompt)
        
        # Test postprocessing logic with dummy text
        dummy_text = "Here is your solution: " + retrieved_solution + " Hope this helps your laptop issue! Let me know if you need more assistance with the wifi."
        clean = clean_output(dummy_text, "FLAN")
        length = enforce_length(clean)
        final = normalize_tone(length)
        
        print("\nMock Postprocessing Results:")
        print(f"Original: {dummy_text[:50]}...")
        print(f"Processed: {final}")

if __name__ == "__main__":
    main()
