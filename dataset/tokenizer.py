from transformers import AutoTokenizer


class NeuroDaptTokenizer:

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
    ):

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    def __call__(
        self,
        paragraph: str,
        target_clause: str,
    ):

        text = (
            f"Paragraph:\n{paragraph}\n\n"
            f"Target Clause:\n{target_clause}"
        )

        return self.tokenizer(
            paragraph,
            target_clause,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )