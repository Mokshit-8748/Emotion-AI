import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


def _load_wavlm_backbone(model_name="microsoft/wavlm-base-plus"):
    try:
        return WavLMModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        print(f"[cache-miss] Falling back to online load for {model_name}")
        return WavLMModel.from_pretrained(model_name)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling: computes attention-weighted mean + std.
    Focuses on emotionally salient frames, ignores silence/noise.
    """
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.conv   = nn.Conv1d(channels, attention_channels, kernel_size=1)
        self.linear = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        x    = x.transpose(1, 2)
        attn = torch.tanh(self.conv(x))
        attn = self.linear(attn)
        alpha = F.softmax(attn, dim=2)          # (B, C, T)
        means = torch.sum(alpha * x, dim=2)     # (B, C)
        residuals = torch.sum(alpha * x**2, dim=2) - means**2
        stds  = torch.sqrt(torch.clamp(residuals, min=1e-9))
        return torch.cat([means, stds], dim=1)  # (B, C*2)


class EmotionWavLM(nn.Module):
    """
    End-to-End Emotion SER Model based on WavLM Base+.

    v2 changes:
    - Head dropout: 0.3/0.2 → 0.4/0.3
    - LayerNorm before each Dense layer in head
    - unfreeze_layers defaults to 6
    - layer_weights initialized to 0 (softmax(0) = uniform start)

    v2.1 changes:
    - layer_weights exposed via get_layer_weight_params() so the optimizer
      can assign it a separate (lower) LR instead of lumping it with the head.
    """
    def __init__(self, num_classes=7, unfreeze_layers=6):
        super().__init__()
        self.wavlm = _load_wavlm_backbone("microsoft/wavlm-base-plus")

        total_layers  = len(self.wavlm.encoder.layers)
        freeze_until  = total_layers - unfreeze_layers

        # Freeze CNN feature extractor always
        self.wavlm.feature_extractor.requires_grad_(False)
        self.wavlm.feature_projection.requires_grad_(False)

        # Selectively freeze/unfreeze transformer layers
        for idx, layer in enumerate(self.wavlm.encoder.layers):
            requires_grad = idx >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

        # FIX v2.1: keep layer_weights as its own attribute so the caller
        # can put it in a separate param group with a controlled LR.
        # softmax(zeros) = uniform, so no layer dominates at init.
        self.layer_weights = nn.Parameter(torch.zeros(total_layers + 1))

        # Attentive Statistics Pooling (768 → 1536)
        self.pooling = AttentiveStatisticsPooling(channels=768)

        # Classification head with LayerNorm + stronger dropout
        self.head = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def get_param_groups(self, lr_backbone, lr_head, lr_layer_weights=None):
        """
        Returns optimizer param groups with correct LRs:
          - WavLM backbone (unfrozen layers only) → lr_backbone
          - layer_weights                         → lr_layer_weights (defaults to lr_backbone)
          - pooling + head                        → lr_head

        Usage:
            optimizer = AdamW(model.get_param_groups(5e-6, 3e-4), weight_decay=0.01)
        """
        if lr_layer_weights is None:
            lr_layer_weights = lr_backbone

        backbone_params     = [p for n, p in self.named_parameters()
                                if p.requires_grad and "wavlm" in n]
        layer_weight_params = [self.layer_weights]
        head_params         = [p for n, p in self.named_parameters()
                                if p.requires_grad and "wavlm" not in n
                                and "layer_weights" not in n]

        return [
            {"params": backbone_params,     "lr": lr_backbone},
            {"params": layer_weight_params, "lr": lr_layer_weights},
            {"params": head_params,         "lr": lr_head},
        ]

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Stack all hidden states: (B, n_layers+1, T, 768)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)

        # Weighted sum across layers
        weights           = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        weighted_features = torch.sum(hidden_states * weights, dim=1)  # (B, T, 768)

        # Attentive pooling → (B, 1536)
        pooled = self.pooling(weighted_features)

        # Classification → (B, num_classes)
        return self.head(pooled)
