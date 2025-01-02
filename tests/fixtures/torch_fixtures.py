import pytest
import torch

__all__ = [
    "torch_device",
    "torch_disable_gradients",
]


@pytest.fixture(autouse=True)
def torch_disable_gradients():
    """Disable gradient tracking globally."""

    torch.set_grad_enabled(False)


@pytest.fixture(scope="session")
def torch_device():
    """Configure gpus."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
