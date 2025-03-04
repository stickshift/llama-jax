import logging

import jax

from . import (
    attention,
    chat,
    checkpoint,
    embeddings,
    ffn,
    head,
    kvc,
    layer,
    model,
    position_mask,
    render,
    rms_norm,
    rope,
    text,
    tools,
)

__all__ = [
    "attention",
    "chat",
    "checkpoint",
    "embeddings",
    "ffn",
    "head",
    "kvc",
    "layer",
    "model",
    "position_mask",
    "render",
    "rms_norm",
    "rope",
    "text",
    "tools",
]


logger = logging.getLogger(__name__)

logger.info(f"Available jax devices: {jax.devices()}")
