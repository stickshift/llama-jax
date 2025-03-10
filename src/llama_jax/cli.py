import pickle

import click
from jax.dlpack import from_dlpack

import llama_jax as ll

__all__ = [
    "cli",
    "convert_checkpoint",
]


@click.group("llama-jax")
def cli() -> None:
    """Llama JAX CLI."""
    pass


@cli.command("convert")
@click.argument(
    "checkpoint",
    type=str,
)
def convert_checkpoint(checkpoint: str) -> None:
    """Convert Llama checkpoint to JAX."""
    import torch  # noqa: PLC0415

    # Load config
    config = ll.checkpoint.load_config(checkpoint)

    # Configure paths
    input_path = config.checkpoint_path / "consolidated.00.pth"
    output_path = config.checkpoint_path / "consolidated.00.jax"

    # Load parameters as torch tensors
    parameters = torch.load(
        input_path,
        weights_only=True,
        map_location="cpu",
    )

    click.echo(f"Read {len(parameters)} parameters from {input_path}")

    # Convert to jax pytree
    parameters = {k: from_dlpack(v) for k, v in parameters.items()}

    # Write parameters using pickle
    output_path.write_bytes(pickle.dumps(parameters))  # noqa

    click.echo(f"Saved {len(parameters)} parameters to {output_path}")
