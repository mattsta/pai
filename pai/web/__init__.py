"""Web interface for pai using HTMX."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .server import create_app, run_server

__all__ = ["create_app", "run_server", "run_server_cli"]

# Create typer app for CLI
app = typer.Typer(
    name="pai-web",
    help="Web interface for pai",
    add_completion=False,
)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8080,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config file"),
    ] = None,
) -> None:
    """Start the pai web interface.

    Runs a web server providing an HTMX-based interface for interacting
    with AI models through pai.
    """
    from ..models import PolyglotConfig
    from ..pricing import PricingService

    # Load config
    if config is None:
        config = Path.home() / ".config" / "pai" / "config.toml"

    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        typer.echo("Please create a config file or specify one with --config", err=True)
        raise typer.Exit(1)

    try:
        toml_config = PolyglotConfig.from_toml(config)
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(1)

    # Initialize pricing service
    pricing_service = PricingService()

    # Run the server
    asyncio.run(run_server(toml_config, pricing_service, host, port))


def run_server_cli() -> None:
    """Entry point for pai-web command."""
    app()


# For direct script execution
if __name__ == "__main__":
    app()
