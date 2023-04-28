import click
import logging
from pathlib import Path
from rich.logging import RichHandler
from typing import Optional


@click.group()
def cli():
    """Perphix CLI"""
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.option("-i", "--input-dir", type=str, required=True, help="Input directory")
@click.option(
    "-o",
    "--output-dir",
    type=str,
    required=True,
    help="Output directory in which 'case-XXXXXX' directories will be created",
)
@click.option(
    "--case",
    type=str,
    default=None,
    help="Case ID. This is used as the new patient identifier. If not provided, a random 6-digit number will be used.",
)
def deidentify(input_dir: Path, output_dir: Path, case: Optional[str] = None):
    """Remove patient identifiable information from DICOM files."""
    from .deidentify import deidentify as _deidentify

    _deidentify(input_dir, output_dir, case)


@cli.command()
@click.option("-i", "--input-dir", type=str, required=True, help="Input directory")
@click.option("-o", "--output-dir", type=str, required=True, help="Output directory")
@click.option("-f", "--format", type=str, default="png", help="Output format. Defaults to 'png'.")
def convert(input_dir: Path, output_dir: Path, format: str = "png"):
    """Convert DICOM files to other formats."""
    from .convert import convert as _convert

    _convert(input_dir, output_dir, format)


if __name__ == "__main__":
    cli()
