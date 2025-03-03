from collections.abc import Sequence
from pathlib import Path

from jax import Array
from jax import numpy as jnp
import tiktoken
from tiktoken.load import load_tiktoken_bpe

import llama_jax as ll
from llama_jax.tools import default_arg

__all__ = [
    "Tokenizer",
]

_special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",  # end of message
    "<|eot_id|>",  # end of turn
    "<|python_tag|>",
    "<|image|>",
]

_num_reserved_special_tokens = 256

_reserved_tokens = [
    f"<|reserved_special_token_{2 + i}|>" for i in range(_num_reserved_special_tokens - len(_special_tokens))
]

_pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501


class Tokenizer:
    """Llama3 tokenizer based on tiktoken and the llama-models implementation."""

    def __init__(self, model_path: Path | str, max_tokens: int):
        # Convert to str
        model_path = str(model_path)

        # Load base tokens from tiktoken model
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(_special_tokens + _reserved_tokens)}

        self.model = tiktoken.Encoding(
            name=model_path,
            pat_str=_pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.max_tokens = max_tokens

        self.n_words: int = num_base_tokens + len(self.special_tokens)
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        self.stop_tokens = jnp.array([
            self.eos_id,
            self.special_tokens["<|eom_id|>"],
            self.special_tokens["<|eot_id|>"],
        ])

    def encode(
        self, prompts: str | Sequence[str], bos: bool | None = None, eos: bool | None = None
    ) -> tuple[Array, Array]:
        """Encodes a list of prompts into an array of token IDs."""
        # Defaults
        bos = default_arg(bos, True)
        eos = default_arg(eos, False)

        # Ensure prompts is a sequence of prompts
        if isinstance(prompts, str):
            prompts = (prompts,)

        # Encode each prompt
        tids = [
            self.model.encode(
                prompt,
                allowed_special="all",
                disallowed_special=(),
            )
            for prompt in prompts
        ]

        # Padding
        pad_length = max(len(v) for v in tids)
        for v in tids:
            v.extend([self.pad_id] * (pad_length - len(v)))

        # Inject bos/eos tokens
        if bos:
            tids = [[self.bos_id, *v] for v in tids]
        if eos:
            tids = [[*v, self.eos_id] for v in tids]

        # Convert to jax
        token_ids = jnp.array(tids)

        # Calculate mask from padding
        position_mask = ll.position_mask.create(jnp.where(token_ids == self.pad_id, 0, 1), max_tokens=self.max_tokens)

        return token_ids, position_mask

    def decode(self, token_ids: Array, special: bool | None = None) -> Sequence[str]:
        """Decodes token_ids into sequence of strings."""
        # Defaults
        special = default_arg(special, True)

        # Validate
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids is not a 2D array: {token_ids.shape}")

        # Collect token ids to decode
        values = [
            [tid.item() for tid in tids if special or tid not in self.special_tokens.values()] for tids in token_ids
        ]

        return tuple(self.model.decode(v) for v in values)
