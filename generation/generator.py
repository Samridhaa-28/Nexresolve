"""
NexResolve Response Generator Core
Manages model loading and response generation logic.
"""
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from .prompt_templates import build_flan_prompt, build_bart_prompt
from .postprocess import clean_output, enforce_length, normalize_tone


class ResponseGenerator:
    def __init__(self, model_type="FLAN", device=None):
        self.model_type = model_type.upper()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "FLAN":
            self.model_id = "google/flan-t5-large"
        elif self.model_type == "BART":
            self.model_id = "facebook/bart-large"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model = None
        self.tokenizer = None
        self._load_all()

    def _load_all(self):
        print(f"Loading {self.model_type} model from {self.model_id} onto {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model_type == "MISTRAL":
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            except Exception as e:
                print(f"Quantization failed, falling back to float16: {str(e)}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        elif self.model_type in ["FLAN", "BART"]:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id
            ).to(self.device)

        self.model.eval()

    def generate(self, ticket_summary, intent, entities, retrieved_solution):
        """
        Generates a clean paragraph response based on the retrieved solution.
        """
        # 1. Build prompt
        if self.model_type == "FLAN":
            prompt = build_flan_prompt(ticket_summary, intent, entities, retrieved_solution)
        elif self.model_type == "BART":
            prompt = build_bart_prompt(ticket_summary, retrieved_solution)

        # 2. Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 4. Decode
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. Postprocess
        final_text = clean_output(decoded_text, self.model_type)
        final_text = enforce_length(final_text)
        final_text = normalize_tone(final_text)

        return final_text