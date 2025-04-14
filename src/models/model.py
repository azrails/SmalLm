import logging
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .config import SmalLmConfig

allow_ops_in_compiled_graph()
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input


logger = logging.getLogger(__name__)


class SwiGLU(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.up_proj = nn.Linear(input_size, hidden_size * 2, bias=bias)
        self.down_proj = nn.Linear(hidden_size, input_size, bias=bias)

    def forward(self, x):
        up_gate = self.up_proj(x)
        up, gate = rearrange(up_gate, "... (d span) -> span ... d", d=self.hidden_size)
        down = F.silu(gate) * up
        return self.down_proj(down)


class Router(nn.Module):
    """
    Router for distribution of tokens by experts in MoE
    """

    def __init__(self, config: SmalLmConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.experts_to_select = self.config.token_experts - self.config.shared_experts
        self.gate = nn.Linear(config.hidden_size, config.routed_experts, bias=False)
        self.gate_noise = (
            nn.Linear(config.hidden_size, config.routed_experts, bias=False)
            if config.noisy_experts is True
            else None
        )
        self.bias_coef = config.balancing_coef
        self.register_buffer(
            "bias", torch.zeros(config.routed_experts), persistent=True
        )
        self.register_buffer(
            "expert_counts", torch.zeros(config.routed_experts), persistent=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # calculating with fp32 for stability
        # num_tokens n_shared_experts
        gate_logits = self.gate(x)
        if self.gate_noise is not None:
            gate_logits_noise = F.softplus(self.gate_noise(x))
            gate_logits_noise = torch.randn_like(gate_logits_noise) * gate_logits_noise
            gate_logits = gate_logits + gate_logits_noise

        gate_weights = gate_logits.sigmoid()
        original_weights = gate_weights

        gate_weights = gate_weights + self.bias

        _, top_experts_idx = torch.topk(gate_weights, self.experts_to_select, dim=-1)
        counts = torch.bincount(
            top_experts_idx.flatten(), minlength=self.config.routed_experts
        ).detach()
        if self.training:
            self.expert_counts += counts
        top_experts_weights = original_weights.gather(1, top_experts_idx)
        top_experts_weights = top_experts_weights / top_experts_weights.sum(
            dim=-1, keepdim=True
        )
        return top_experts_idx, top_experts_weights.type_as(x), counts.tolist()

    def update_bias(self):
        mean = self.expert_counts.float().mean()
        delta = self.bias_coef * torch.sign(mean - self.expert_counts)
        self.bias += delta
        self.expert_counts.zero_()


class MoE(nn.Module):
    """
    MoE experts, contains shared and routed experts,
    like DeepSeek MoE, also use Auxiliary-Loss-Free Load Balancing
    ref: https://arxiv.org/abs/2408.15664
    """

    def __init__(self, config: SmalLmConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.shared_experts = SwiGLU(
            config.hidden_size,
            config.shared_experts * config.expert_size,
            config.moe_bias,
        )
        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.expert_size, config.moe_bias)
                for _ in range(config.routed_experts)
            ]
        )
        self.router = Router(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.config.hidden_size)
        experts_idx, experts_weights, counts = self.router(x)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.routed_experts):
            if counts[i] == 0:
                continue
            idx, pos = torch.where(experts_idx == i)
            out[idx] += expert(x[idx]) * experts_weights[idx, pos, None]
        shared_out = self.shared_experts(x)
        return (out + shared_out).view(shape)


def build_alibi_bias(config: SmalLmConfig) -> torch.Tensor:
    """
    Build ALiBi bias for specified number of heads
    ref: https://arxiv.org/abs/2108.12409v2

    Returns:
        Tensor with ALiBi biases, shape: [num heads]
    """
    bias = (
        2**-8
        / config.num_attention_heads
        * torch.arange(1, config.num_attention_heads + 1).float()
    )
    return bias


def calc_rotation(num_rotaitions, dim, base, seq_len) -> torch.Tensor:
    """
    In terms of wavelength calculate the position for a specific rotation frequence
    """
    return (
        dim
        * torch.log(torch.tensor(seq_len).float() / (num_rotaitions * 2 * torch.pi))
        / torch.log(torch.tensor(base))
    )


def get_ramp_interpolation(min_idx, max_idx, thetas_dim, eps=1e-6) -> torch.Tensor:
    """
    Ramp interpolation function to maintain high frequencies and expand low frequencies
    """
    if min_idx == max_idx:
        max_idx += eps
    mult = (torch.arange(thetas_dim) - min_idx) / (max_idx - min_idx)
    mult = torch.clamp(mult, 0, 1)
    return 1 - mult


def build_rope_bias(config: SmalLmConfig) -> torch.Tensor:
    """
    Build RoPE bias for specified dimension and maximum sequence length
    uses complex space for simplicity and convenience
    ref: https://arxiv.org/abs/2104.09864v5
    Also use NTK-by-parts interpolation method
    ref: https://arxiv.org/abs/2309.00071
    good explanation: https://blog.eleuther.ai/yarn/

    Args:
        config (SmalLmConfig): base model config

    Returns:
        torch.Tensor: Complex values for rotations, shape: [seq_len, head_size]
    """
    dim = config.head_size

    theta = 1.0 / (config.rope_base ** (torch.arange(0, dim, 2).float() / dim))

    # neural tangent kernel by part korrection
    if config.max_seq_len > config.original_seq_len:
        scale = config.max_seq_len / config.original_seq_len
        # from idea that lambda = 2pi / theta_i and also lambda = seq_len / num_rotations, lambda - wavelen
        low_interpolation_idx = max(
            0,
            torch.ceil(
                calc_rotation(
                    config.high_rotations,
                    dim,
                    config.rope_base,
                    config.original_seq_len,
                )
            ).item(),
        )
        high_interpolation_idx = min(
            dim - 1,
            torch.floor(
                calc_rotation(
                    config.low_rotations, dim, config.rope_base, config.original_seq_len
                )
            ).item(),
        )
        interpolation_mult = get_ramp_interpolation(
            low_interpolation_idx, high_interpolation_idx, dim // 2
        )
        theta = (1 - interpolation_mult) * theta / scale + interpolation_mult * theta

    seq_idx = torch.arange(config.max_seq_len)
    seq_theta = torch.outer(seq_idx, theta)
    bias = torch.polar(torch.ones_like(seq_theta), seq_theta)
    return bias


def apply_rope_bias(x: torch.Tensor, precompute_bias: torch.Tensor) -> torch.Tensor:
    """
    Apply rope bias in complex space

    Args:
        x (torch.Tensor): input embeddings for head
        precompute_bias (torch.Tensor): precomputed rope bias

    Returns:
        torch.Tensor: rotated embeddings
    """
    ini_dtype = x.dtype
    # for numerical stability convert to fp32
    x = rearrange(x.float(), "b n s (d i) -> b n s d i", i=2).contiguous()
    x = torch.view_as_complex(x)
    x = x * precompute_bias
    x = torch.view_as_real(x)
    x = rearrange(x, "b n s d i -> b n s (d i)")
    return x.to(ini_dtype)


def flash_attention_forward(
    module: nn.Module,
    x: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    alibi_slope: Optional[torch.Tensor],
) -> torch.Tensor:
    query = rearrange(query, "b n s d -> b s n d")
    key = rearrange(key, "b n s d -> b s n d")
    value = rearrange(value, "b n s d -> b s n d")
    query, idx_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(query, attention_mask)
    key, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(key, attention_mask)
    value, _, _, _, _ = unpad_input(value, attention_mask)

    key = key.contiguous()
    value = value.contiguous()
    query = query.contiguous()

    attention_probs = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=module.config.attention_dropout if module.training else 0.0,
        causal=True,
        alibi_slopes=alibi_slope if module.config.attention_bias == "alibi" else None,
    )
    attention_probs = pad_input(attention_probs, idx_q, x.size(0), x.size(1))
    out = rearrange(attention_probs, "b s n d -> b s (n d)")
    return out, None


def sdpa_attention_forward(
    module: nn.Module,
    x: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    alibi_slope: Optional[torch.Tensor],
) -> torch.Tensor:
    is_causal = attention_mask is None and query.size(-2) > 1

    attention_probs = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        enable_gqa=True,
        is_causal=is_causal,
        dropout_p=module.config.attention_dropout if module.training else 0.0,
    )
    out = rearrange(attention_probs, "b n s d -> b s (n d)")

    return out, None


def eager_attention_forward(
    module: nn.Module,
    x: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    alibi_slope: Optional[torch.Tensor],
) -> torch.Tensor:
    query = rearrange(
        query,
        "b (kv group) s d -> b kv group s d",
        kv=module.config.num_kv_heads,
        group=module.head_per_group,
    )
    key = rearrange(key, "b kv s d -> b kv 1 s d")
    value = rearrange(value, "b kv s d -> b kv 1 s d")
    attention_weights = query @ key.transpose(-1, -2)
    attention_probs = F.dropout(
        attention_weights / torch.sqrt(torch.tensor(value.size(-1), device=x.device)),
        p=module.config.attention_dropout if module.training else 0.0,
    )
    if alibi_slope is not None:
        alibi_slope = rearrange(
            alibi_slope,
            "b n s s -> b kv group s s",
            kv=module.config.num_kv_heads,
            group=module.head_per_group,
        )
        attention_probs = attention_probs + alibi_slope
    elif alibi_slope is None and attention_mask is not None:
        attention_mask = attention_mask.expand(
            -1, module.config.num_attention_heads, -1, -1
        )
        attention_mask = rearrange(
            attention_mask,
            "b (kv group) s1 s2 -> b kv group s1 s2",
            kv=module.config.num_kv_heads,
            group=module.head_per_group,
        )
        attention_probs = attention_probs + attention_mask
    attention_probs = F.softmax(attention_probs, dim=-1)
    attention_probs = attention_probs @ value
    out = rearrange(attention_probs, "b kv group s d -> b s (kv group d)")
    return out, attention_weights


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
    "flash_attention_2": flash_attention_forward,
}


class CausalSelfAttention(nn.Module):
    """
    Scaled dot product attention with supports different implementations
    currently available: sdpa, flash, native torch
    """

    def __init__(self, config: SmalLmConfig, layer_idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.num_attention_heads % config.num_kv_heads != 0:
            raise ValueError("Num attention heads should divided by num kv heads")

        self.config = config
        self.layer_idx = layer_idx
        self.head_per_group = config.num_attention_heads // config.num_kv_heads
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.head_size * config.num_attention_heads,
            bias=config.attention_bias,
        )
        self.kv_proj = nn.Linear(
            config.hidden_size,
            config.head_size * config.num_kv_heads * 2,
            bias=config.attention_bias,
        )
        self.out_proj = nn.Linear(
            config.head_size * config.num_attention_heads,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Cache | torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
        bias: torch.Tensor,
    ):
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.config.num_attention_heads)
        k, v = rearrange(kv, "b s (n d q) -> q b n s d", q=2, d=self.config.head_size)

        if self.config.positional_bias_type == "rope":
            k = apply_rope_bias(k, bias)
            q = apply_rope_bias(q, bias)

        if past_key_values is not None:
            # for static cache
            cach_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(
                key_states=k,
                value_states=v,
                layer_idx=self.layer_idx,
                cache_kwargs=cach_kwargs,
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        out, attention_weights = attention_interface(
            self,
            x,
            q,
            k,
            v,
            attention_mask,
            bias if self.config.positional_bias_type == "alibi" else None,
        )

        out = self.out_proj(out)
        return out, attention_weights


class WeightedResidual(nn.Module):
    """
    Weighted residual connection, possibly learn skip weight
    """

    def __init__(self, config: SmalLmConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size), requires_grad=config.static_residual
        )

    def forward(self, short, long):
        return self.weight * short + long


class Block(nn.Module):
    def __init__(self, config: SmalLmConfig, layer_idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=config.rms_affine,
        )
        self.ffn_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=config.rms_affine,
        )
        self.dropout1 = nn.Dropout(config.layer_dropout)
        self.dropout2 = nn.Dropout(config.layer_dropout)
        self.attention = CausalSelfAttention(config, layer_idx)
        self.mlp = (
            MoE(config)
            if (
                config.use_moe
                and layer_idx % config.moe_period == 0
                and layer_idx > config.no_moe_layers
            )
            else SwiGLU(config.hidden_size, config.intermediate_size, config.mlp_bias)
        )
        self.attention_residual = WeightedResidual(config)
        self.ffn_residual = WeightedResidual(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Cache | torch.FloatTensor],
        output_attentions: bool,
        cache_position: Optional[torch.LongTensor],
        bias: torch.Tensor,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        identity = inputs_embeds

        # attention block
        out = self.attn_norm(inputs_embeds)
        out, attention_probs = self.attention(
            out, attention_mask, past_key_values, cache_position, bias
        )
        out = self.dropout1(out)
        identity = self.attention_residual(identity, out)

        # swiglu / MoE block
        out = self.dropout2(self.mlp(self.ffn_norm(identity)))
        out = self.ffn_residual(identity, out)
        if output_attentions:
            return out, attention_probs
        return (out,)


class SmalLmPreTrainedModel(PreTrainedModel):
    config_class = SmalLmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            module.weight.data[self.pad_idx].zero_()


class SmalLmModel(SmalLmPreTrainedModel):
    def __init__(self, config: SmalLmConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.pad_idx = config.pad_token_id
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        precompute_bias = (
            build_alibi_bias(config)
            if config.positional_bias_type == "alibi"
            else build_rope_bias(config)
        )
        self.register_buffer("precompute_bias", precompute_bias, persistent=False)
        # не забыть про sharing weights на output голове self.embedding.weight = self.output.weight
        self.embedding = nn.Embedding(
            self.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList(
            [Block(config, idx) for idx in range(1, config.num_hidden_layers + 1)]
        )
        self.out_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=config.rms_affine,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        # input options
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # output options
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache options
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache | torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        # check additional parameters
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (False if self.training else self.config.use_cache)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You must specify only input_ids or inputs_embeds, not both"
            )

        if self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # calculating position for StaticCache
        if cache_position is None:
            last_position = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                last_position,
                last_position + inputs_embeds.size(1),
                device=inputs_embeds.device,
            )

        causal_mask = self._get_causal_masks(
            attention_mask, inputs_embeds, past_key_values, cache_position
        )
        if self.config.positional_bias_type == "rope":
            end_pos = (
                inputs_embeds.size(1)
                if past_key_values is None
                else cache_position[-1] + 1
            )
            start_pos = 0 if past_key_values is None else cache_position[0]
            bias = self.precompute_bias[start_pos:end_pos]

        elif self.config.positional_bias_type == "alibi":
            if self.config._attn_implementation == "flash_attention_2":
                bias = self.precompute_bias
            else:
                i = torch.arange(
                    (
                        inputs_embeds.size(1)
                        if past_key_values is None
                        else cache_position[-1] + 1
                    ),
                    device=inputs_embeds.device,
                )
                bias = i[:, None] - i[None, :]
                bias = torch.tril(bias).expand(
                    inputs_embeds.size(0), self.config.num_attention_heads, -1, -1
                ) * rearrange(self.precompute_bias, "n -> 1 n 1 1")
                if causal_mask is not None:
                    causal_mask = causal_mask + bias
                else:
                    causal_mask = bias

        hidden_state = inputs_embeds
        hidden_states = [hidden_state] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for idx, layer in enumerate(self.layers, 1):
            if self.gradient_checkpointing:
                # for details see:
                # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3107
                # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3149
                layer_out = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_state,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    cache_position,
                    bias,
                )
            else:
                layer_out = layer(
                    hidden_state,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    cache_position,
                    bias,
                )
            hidden_state = layer_out[0]
            if output_hidden_states:
                hidden_states.append(hidden_state)
            if output_attentions:
                attentions.append(layer_out[1])

        hidden_state = self.out_norm(hidden_state)
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_state,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )
        return out if return_dict else out.to_tuple()

    def _get_causal_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is None:
                attention_mask = torch.ones(
                    (inputs_embeds.size(0), inputs_embeds.size(1)),
                    device=inputs_embeds.device,
                ).long()
            return attention_mask
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        past_token = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if attention_mask is not None and torch.all(attention_mask == 0.0):
            return None
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_token,
            is_training=self.training,
        ):
            return None

        sequence_length = inputs_embeds.size(1)
        target_length = (
            attention_mask.size(-1)
            if isinstance(attention_mask, torch.Tensor)
            else past_token + sequence_length + 1
        )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=inputs_embeds.size(0),
        )

        min_dtype = torch.finfo(dtype).min
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: Optional[torch.Tensor],
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class SmalLmForCausalLM(SmalLmPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SmalLmConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.model = SmalLmModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        # input options
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # output options
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache options
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache | torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # generation options
        labels: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = model_outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits, model_outputs[1:])
            return (loss, output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


__all__ = ["SmalLmForCausalLM", "SmalLmModel", "SmalLmPreTrainedModel"]
