"""Tiny language models for UABLA experiments and baselines."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lets non-torch config tests import the package.
    torch = None
    F = None
    nn = object

from .config import UABLAConfig

if torch is not None:
    from .attention import UABLAAttention, UABLAOutput
else:  # pragma: no cover
    UABLAAttention = object
    UABLAOutput = object


@dataclass(frozen=True)
class DenseAttentionOutput:
    output: torch.Tensor
    attention: torch.Tensor | None = None


@dataclass(frozen=True)
class TinyLMOutput:
    logits: torch.Tensor
    uabla_outputs: tuple[UABLAOutput, ...] = ()
    dense_attentions: tuple[torch.Tensor, ...] = ()


class DenseCausalSelfAttention(nn.Module):
    """Small single-head dense/local causal attention baseline."""

    def __init__(self, hidden_size: int, *, local_window: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.local_window = local_window
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, return_attention: bool = False) -> torch.Tensor | DenseAttentionOutput:
        batch, seq_len, hidden_size = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = q @ k.transpose(-1, -2) / (hidden_size**0.5)
        positions = torch.arange(seq_len, device=x.device)
        causal_mask = positions.view(seq_len, 1) >= positions.view(1, seq_len)
        if self.local_window is not None:
            local_mask = positions.view(seq_len, 1) - positions.view(1, seq_len) < self.local_window
            causal_mask = causal_mask & local_mask
        scores = scores.masked_fill(~causal_mask.view(1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = self.out(attention @ v)
        if return_attention:
            return DenseAttentionOutput(output=output, attention=attention)
        return output


class CausalConvMixer(nn.Module):
    """Cheap local byte/token mixer used before global attention.

    The convolution is left-padded only, so token representations never see
    future positions before the causal transformer stack.
    """

    def __init__(self, hidden_size: int, *, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size <= 1:
            raise ValueError("kernel_size must be greater than 1")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=False,
        )
        self.pointwise = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = F.pad(y, (self.kernel_size - 1, 0))
        y = self.depthwise(y).transpose(1, 2)
        y = self.pointwise(F.gelu(y))
        return self.norm(x + self.dropout(y))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_ratio: int,
        attention_type: str,
        dropout: float,
        local_window: int,
        uabla_config: UABLAConfig | None,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.norm1 = nn.LayerNorm(hidden_size)
        if attention_type == "uabla":
            if uabla_config is None:
                raise ValueError("uabla_config is required for UABLA attention")
            self.attention = UABLAAttention(uabla_config)
        elif attention_type == "dense":
            self.attention = DenseCausalSelfAttention(hidden_size, dropout=dropout)
        elif attention_type == "local":
            self.attention = DenseCausalSelfAttention(
                hidden_size,
                local_window=local_window,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown attention_type: {attention_type}")
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_routing: bool = False,
        return_attention: bool = False,
        return_token_scores: bool = False,
    ) -> tuple[torch.Tensor, UABLAOutput | None, torch.Tensor | None]:
        attention_input = self.norm1(x)
        uabla_output = None
        dense_attention = None
        if self.attention_type == "uabla":
            attention_result = self.attention(
                attention_input,
                return_routing=return_routing,
                return_attention=return_attention,
                return_token_scores=return_token_scores,
            )
            if isinstance(attention_result, UABLAOutput):
                attention_output = attention_result.output
                uabla_output = attention_result
            else:
                attention_output = attention_result
        else:
            attention_result = self.attention(attention_input, return_attention=return_attention)
            if isinstance(attention_result, DenseAttentionOutput):
                attention_output = attention_result.output
                dense_attention = attention_result.attention
            else:
                attention_output = attention_result
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x, uabla_output, dense_attention


class TinyTransformerLM(nn.Module):
    """Tiny LM wrapper for synthetic UABLA comparisons."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        attention_type: str = "uabla",
        local_window: int = 64,
        dropout: float = 0.0,
        uabla_config: UABLAConfig | None = None,
        uabla_vectorized_routing: bool = True,
        input_mixer_kernel: int = 0,
    ) -> None:
        super().__init__()
        if torch is None:
            raise ModuleNotFoundError("TinyTransformerLM requires torch")
        if input_mixer_kernel < 0:
            raise ValueError("input_mixer_kernel cannot be negative")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        if attention_type == "uabla" and uabla_config is None:
            uabla_config = UABLAConfig(
                hidden_size=hidden_size,
                routing_dim=max(8, hidden_size // 4),
                value_dim=max(16, hidden_size // 2),
                position_dim=max(4, hidden_size // 8),
                block_size=16,
                local_window=local_window,
                centroid_hit_buckets=(4, 8),
                token_k_buckets=(8, 16),
                superblock_size_blocks=4,
                superblock_hit_buckets=(2, 4),
                vectorized_routing=uabla_vectorized_routing,
            )
        self.uabla_config = uabla_config
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_mixer = (
            CausalConvMixer(hidden_size, kernel_size=input_mixer_kernel, dropout=dropout)
            if input_mixer_kernel > 1
            else None
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_ratio=mlp_ratio,
                    attention_type=attention_type,
                    dropout=dropout,
                    local_window=local_window,
                    uabla_config=uabla_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_routing: bool = False,
        return_attention: bool = False,
        return_token_scores: bool = False,
    ) -> TinyLMOutput:
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).view(1, seq_len)
        x = self.token_embedding(input_ids)
        if self.input_mixer is not None:
            x = self.input_mixer(x)
        x = x + self.position_embedding(positions)
        x = self.drop(x)
        uabla_outputs: list[UABLAOutput] = []
        dense_attentions: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            x, uabla_output, dense_attention = block(
                x,
                return_routing=return_routing,
                return_attention=return_attention,
                return_token_scores=return_token_scores and block_idx == len(self.blocks) - 1,
            )
            if uabla_output is not None:
                uabla_outputs.append(uabla_output)
            if dense_attention is not None:
                dense_attentions.append(dense_attention)
        logits = self.lm_head(self.norm(x))
        return TinyLMOutput(
            logits=logits,
            uabla_outputs=tuple(uabla_outputs),
            dense_attentions=tuple(dense_attentions),
        )

    @property
    def cache_dim_per_token_per_layer(self) -> int:
        if self.attention_type == "uabla":
            if self.uabla_config is None:
                raise RuntimeError("uabla_config missing")
            return self.uabla_config.cache_dim
        return 2 * self.hidden_size
