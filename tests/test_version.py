"""Keep package and CLI versions aligned."""

from click.testing import CliRunner

from rag import __version__
from rag.cli import cli


def test_version_consistency() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"rag-eval, version {__version__}"
