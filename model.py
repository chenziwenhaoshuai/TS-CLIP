import torch
import numpy as np
from torch import nn
from time_moe.models.modeling_time_moe import TimeMoeModel


class TS_CLIP(nn.Module):
    """
    TS-CLIP: A contrastive learning framework that aligns time series representations
    with natural language descriptions using a TimeMoE encoder and a linear projector.

    The model is initialised with the TimeMoE architecture from `model_path` (used only
    for the config / architecture), and all weights are loaded from the TS-CLIP checkpoint
    by the caller — no separate TimeMoE pre-trained weights file is needed at inference.
    """

    def __init__(self, model_path: str = './TimeMoE_50M'):
        super(TS_CLIP, self).__init__()
        # Build the architecture from config only; weights will be loaded via checkpoint.
        self.Time_encoder = TimeMoeModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.ts_projector = nn.Linear(384, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ts, text_embedding):
        attention_mask = torch.ones(ts.size(0), ts.size(1), device=ts.device)
        position_ids = torch.arange(ts.size(1), device=ts.device).unsqueeze(0).expand(ts.size(0), -1)
        ts_embedding = self.Time_encoder(
            ts, attention_mask=attention_mask, position_ids=position_ids
        ).last_hidden_state.mean(dim=1)
        ts_embedding = self.ts_projector(ts_embedding)
        return ts_embedding, text_embedding
