"""
This file contains the model architecture: context encoder, target encoder, and predictor.
"""

import warnings

import torch
from torch import nn

from utils.config import ModelConfig


def _largest_head_count_at_most(max_heads: int, d_model: int) -> int:
    """Pick the largest nhead <= max_heads that divides d_model (Transformer requirement)."""
    for h in range(min(max_heads, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


class TokenEmbedder(nn.Module):
    """
    Project token ids into a latent space.

    In a real JEPA system, this would be a richer patch/token embedding stage.
    For logs, the equivalent is a learned embedding table plus positional encoding.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x))


class FeedForwardBlock(nn.Module):
    """
    Small feed-forward after the transformer layers.

    Gives the model a chance to shape the latent space before the predictor sees it, helping the latent space be better for JEPA
    """
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogSequenceEncoder(nn.Module):
    """
    Deep sequence encoder for log lines.

    This is the part that turns tokenized log lines into contextual embeddings.
    It is intentionally stronger than a simple linear layer because JEPA relies on
    good latent representations on both the context and target sides.

    Token vectors are learned, and positional encoding is also learned.
    """

    def __init__(self, vocab_size: int, latent_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.token_embedder = TokenEmbedder(vocab_size=vocab_size, embed_dim=latent_dim)
        self.line_proj = nn.Linear(latent_dim, latent_dim)
        self.token_pos_embed = nn.Parameter(torch.zeros(1, 128, latent_dim))
        self.line_pos_embed = nn.Parameter(torch.zeros(1, 1024, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.line_transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.ffn = FeedForwardBlock(latent_dim, latent_dim * 4, dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x has shape: [batch, lines, tokens_per_line]
        batch_size, num_lines, num_tokens = x.shape
        x = self.token_embedder(x)
        token_pos = self.token_pos_embed[:, :num_tokens, :].unsqueeze(1)
        x = x + token_pos
        # collapse all token embeddings into a single embedding for the whole log line
        x = x.mean(dim=2)
        x = self.line_proj(x)
        x = x + self.line_pos_embed[:, :num_lines, :]

        if mask is not None:
            # mask marks which log lines are visible - keep the same batch shape and zero-out hidden lines so downstream shapes are stable
            if mask.dim() == 2:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            elif mask.dim() == 1:
                x = x[:, mask, :]

        x = self.line_transformer(x)
        x = self.ffn(x)
        return self.norm(x)


class Predictor(nn.Module):
    """
    The predictor takes the context representation and tries to guess the latent vectors for the masked/hidden target tokens.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, latent_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, context_tokens: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        batch_size, _, dim = context_tokens.shape
        # convert target positions into conditioning information - position embeddings tell the predictor which hidden region it should generate
        pos = self.pos_embed[:, target_positions, :].expand(batch_size, -1, -1)
        # mask tokens represent the missing content.
        pred_tokens = self.mask_token.expand(batch_size, target_positions.numel(), dim) + pos
        # predictor sees the context tokens plus placeholder tokens for what is missing
        x = torch.cat([context_tokens, pred_tokens], dim=1)
        x = self.blocks(x)
        x = self.norm(x[:, -target_positions.numel() :, :])
        return self.proj(x) * self.output_scale


class JEPAArchitecture(nn.Module):
    """
    Full JEPA architecture - context encoder, target encoder, and predictor.
    """

    def __init__(self, model_config: ModelConfig, vocab_size: int):
        super().__init__()
        latent_dim = model_config.latent_dim
        requested_hidden = model_config.predictor.hidden_dim
        dropout = model_config.encoder.dropout
        enc_layers = max(1, model_config.encoder.num_layers)
        enc_heads = max(1, model_config.encoder.num_heads)
        encoder_ffn = latent_dim * 4  # matches LogSequenceEncoder TransformerEncoderLayer dim_feedforward

        predictor_layers = getattr(model_config.predictor, "num_layers", 2)
        predictor_heads = getattr(model_config.predictor, "num_heads", 4)

        # Predictor is strictly smaller than the encoder (depth, width, heads), matching JEPA / I-JEPA inductive bias.
        pred_layers = min(predictor_layers, max(1, enc_layers - 1))
        pred_ffn = min(max(requested_hidden, latent_dim), encoder_ffn - 1)
        head_cap = enc_heads - 1 if enc_heads > 1 else 1
        pred_heads = _largest_head_count_at_most(min(predictor_heads, head_cap), latent_dim)

        if pred_layers != predictor_layers or pred_ffn != requested_hidden or pred_heads != predictor_heads:
            warnings.warn(
                "Predictor config clamped to stay strictly smaller than the encoder: "
                f"layers {predictor_layers}->{pred_layers}, "
                f"dim_feedforward {requested_hidden}->{pred_ffn} (encoder uses {encoder_ffn}), "
                f"heads {predictor_heads}->{pred_heads} (encoder uses {enc_heads}).",
                stacklevel=2,
            )

        self.context_encoder = LogSequenceEncoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=enc_layers,
            num_heads=enc_heads,
            dropout=dropout,
        )
        self.target_encoder = LogSequenceEncoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=enc_layers,
            num_heads=enc_heads,
            dropout=dropout,
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.predictor = Predictor(
            latent_dim,
            pred_ffn,
            dropout,
            num_layers=pred_layers,
            num_heads=pred_heads,
        )

        # target encoder is a moving-average teacher - it does not receive gradient updates directly
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def encode_context(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.context_encoder(x, mask=mask)

    def encode_target(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.target_encoder(x, mask=mask)

    def predict(self, context_latent: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_latent, target_positions)

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float) -> None:
        """
        Update the target encoder w/ EMA.
        This is the V-JEPA style stabilizer that keeps the target branch from collapsing.
        """

        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(ema_decay).add_((1.0 - ema_decay) * context_param.data)
