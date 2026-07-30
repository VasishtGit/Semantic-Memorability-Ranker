import torch
import torch.nn as nn


class ClauseAttentionPooling(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
    ):
        super().__init__()

        self.score = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        """
        hidden_states:
            (B,T,H)

        attention_mask:
            (B,T)
        """

        scores = self.score(
            hidden_states,
        ).squeeze(-1)

        scores = scores.masked_fill(
            attention_mask == 0,
            torch.finfo(scores.dtype).min,
        )   

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            hidden_states * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled