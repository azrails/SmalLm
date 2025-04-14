import logging
from typing import Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class SmalLmConfig(PretrainedConfig):
    """
    Base config for all SmalLm models

    Raises:
        ValueError: Positional_bias_type must be in suported types
        ValueError: In case of rope positional_bias_type head_size can't be anything
    """

    model_type = "smallm"

    def __init__(
        self,
        # global model params
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        mlp_bias: bool = False,
        num_hidden_layers: int = 27,
        rms_norm_eps: float = 1e-6,
        rms_affine: bool = False,
        initializer_range: float = 0.02,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = True,
        sliding_window_attention: bool = True,
        sliding_window_context: int = 1024,
        sliding_window_period: int = 4,
        embedding_dropout: float = 0.0,
        layer_dropout: float = 0.1,
        max_seq_len: int = 2048,
        original_seq_len: int | None = None,
        tie_word_embeddings: bool = True,
        # attention params
        num_attention_heads: int = 9,
        num_kv_heads: int = 3,
        head_size: Optional[int] = None,
        attention_dropout: float = 0.1,
        positional_bias_type: str = "rope",
        high_rotations: int = 32,
        low_rotations: int = 1,
        attention_bias: bool = False,
        rope_base: int = 100000,
        # MoE params
        use_moe: bool = True,
        moe_period: int = 3,
        expert_size: int = 256,
        shared_experts: int = 2,
        routed_experts: int = 16,
        token_experts: int = 4,
        noisy_experts: bool = False,
        moe_bias: bool = False,
        balancing_coef: float = 1e-4,
        no_moe_layers: int = 5,
        # extra params
        vocab_size: int = 60000,
        bos_token_id: int = 1,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        static_residual: bool = False,
        **kwargs,
    ):
        if positional_bias_type not in ["alibi", "rope"]:
            raise ValueError(
                f"positional_bias_type must be 'alibi' or 'rope', got {positional_bias_type}"
            )
        self.static_residual = not static_residual
        self.no_moe_layers = no_moe_layers
        self.moe_bias = moe_bias
        self.balancing_coef = balancing_coef
        self.noisy_experts = noisy_experts
        self.high_rotations = high_rotations
        self.low_rotations = low_rotations
        self.positional_bias_type = positional_bias_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_bias = mlp_bias
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.embedding_dropout = embedding_dropout
        self.rms_affine = rms_affine
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.layer_dropout = layer_dropout
        self.use_moe = use_moe
        self.moe_period = moe_period
        self.expert_size = expert_size
        self.shared_experts = shared_experts
        self.routed_experts = routed_experts
        self.token_experts = token_experts
        self.intermediate_size = intermediate_size
        self.attention_bias = attention_bias
        self.rope_base = rope_base
        self.head_size = head_size if head_size else hidden_size // num_attention_heads
        self.original_seq_len = (
            original_seq_len if original_seq_len is not None else max_seq_len
        )

        self.sliding_window_attention = sliding_window_attention
        self.sliding_window_context = sliding_window_context
        self.sliding_window_period = sliding_window_period
        if sliding_window_attention and sliding_window_context > max_seq_len:
            logger.warning(
                f"sliding_window_context more than max_seq_len, \
                    set sliding_window_context to {max_seq_len}"
            )
            self.sliding_window_context = max_seq_len
        if not sliding_window_attention:
            self.sliding_window_context = max_seq_len

        if self.head_size % 2 != 0 and self.positional_bias_type == "rope":
            raise ValueError("Head size should divided by 2")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["SmalLmConfig"]
