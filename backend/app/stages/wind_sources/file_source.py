"""
FileWindSource — the original bundled wind.json behaviour, unchanged.

This preserves the exact pre-interface semantics: it does the same
`json.load(open(path, encoding="utf-8"))` main.py's `_load_json` did, returns
the list of samples verbatim (same values, same units, same timestamps, same
ordering), and adds nothing to them. The pipeline's gate decisions, rejection
reasons and scene behaviour are therefore identical to before — byte-for-byte.
"""

from __future__ import annotations

import json

from . import WindSource


class FileWindSource(WindSource):
    name = "file"

    def __init__(self, path=None):
        super().__init__()
        # An explicit path pins this source to one file; otherwise the per-scene
        # wind.json path passed to fetch() is used (the normal pipeline case).
        self._path = path

    def fetch(self, scene, wind_path) -> list[dict]:
        path = self._path if self._path is not None else wind_path
        # Identical to main.py._load_json — do not "improve" it: byte-identity.
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        self.provenance = {
            "source": "FILE",
            "file": str(path),
            "n_samples": len(samples),
            "cache": "n/a",
        }
        return samples
