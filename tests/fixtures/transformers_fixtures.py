import pytest
from transformers import AutoModelForCausalLM

__all__ = [
    "transformers_checkpoint",
    "transformers_model",
]


@pytest.fixture(scope="session")
def transformers_checkpoint() -> str:
    """Transformers checkpoint for llama3.2-3b."""
    return "meta-llama/llama-3.2-3B"


@pytest.fixture(scope="session")
def transformers_model(torch_device, transformers_checkpoint: str):
    """Transformers model for llama3.2-3b."""

    return AutoModelForCausalLM.from_pretrained(transformers_checkpoint).to(torch_device).eval()
