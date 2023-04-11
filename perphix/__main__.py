import click
import logging
from rich.logging import RichHandler


@click.group()
def cli():
    """Perphix CLI"""
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )