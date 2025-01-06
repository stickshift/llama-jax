import pytest

from llama_jax.benchmarks import llama_models as llm
from llama_jax.benchmarks.llama_models import Transformer
from llama_jax.checkpoint import ModelConfig

__all__ = [
    "reference_model",
]


@pytest.fixture(scope="session")
def reference_model(config: ModelConfig, torch_device):
    """Reference model for llama3.2-3b."""

    transformer = Transformer(config, torch_device)
    transformer.load_state_dict(llm.load_parameters(config, device=torch_device))

    return transformer
