"""Hermes plugin source shipped with Khala.

`khala install hermes` copies this directory to `~/.hermes/plugins/khala/`
and enables it. The plugin mirrors the `khala hook hermes` shell hooks via
Hermes lifecycle plugin hooks, which — unlike shell hooks — fire in every
process that runs turns, including the stdio TUI backend.
"""

from pathlib import Path

PLUGIN_NAME = "khala"
PLUGIN_SOURCE_DIR = Path(__file__).resolve().parent / "hermes"
HERMES_PLUGINS_DIR = Path("~/.hermes/plugins").expanduser()
