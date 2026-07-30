"""ModernBERT backbone wrapper used as the feature extractor."""

from transformers import AutoModel
import torch.nn as nn


class Backbone(nn.Module):

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n_layers > 0:

            for layer in self.encoder.layers[-unfreeze_last_n_layers:]:

                for param in layer.parameters():
                    param.requires_grad = True

            # Also unfreeze the final LayerNorm
            for param in self.encoder.final_norm.parameters():
                param.requires_grad = True

        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state