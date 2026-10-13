"""Efficient lookup of full chunk text by chunk ID from a JSONL corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class CorpusTextStore:
    """Read selected chunks without loading the full corpus into memory."""

    def __init__(self, corpus_path: str | Path, offsets_path: str | Path | None = None):
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"Chunk corpus not found: {self.corpus_path}")

        self.offsets_path = (
            Path(offsets_path)
            if offsets_path
            else self.corpus_path.with_suffix(self.corpus_path.suffix + ".offsets.json")
        )
        self._offsets: dict[str, list[int]] | None = None

    def _build_offsets(self) -> dict[str, list[int]]:
        offsets: dict[str, list[int]] = {}
        with self.corpus_path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid JSONL record at byte offset {offset}"
                    ) from exc

                chunk_id = record.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError(
                        f"Missing chunk_id at byte offset {offset}"
                    )
                if chunk_id in offsets:
                    raise ValueError(f"Duplicate chunk_id in corpus: {chunk_id}")
                offsets[chunk_id] = [offset, len(line)]

        self.offsets_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.offsets_path.with_suffix(self.offsets_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as target:
            json.dump(offsets, target, separators=(",", ":"))
        temp_path.replace(self.offsets_path)
        return offsets

    def _load_offsets(self) -> dict[str, list[int]]:
        if self._offsets is not None:
            return self._offsets

        if self.offsets_path.is_file():
            try:
                with self.offsets_path.open("r", encoding="utf-8") as source:
                    offsets = json.load(source)
                if not isinstance(offsets, dict):
                    raise ValueError("Offset index is not a mapping.")
                self._offsets = offsets
                return offsets
            except (OSError, json.JSONDecodeError, ValueError):
                # Rebuild a damaged or unreadable sidecar.
                pass

        self._offsets = self._build_offsets()
        return self._offsets

    def get_many(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return full records for requested IDs; unknown IDs are omitted."""
        offsets = self._load_offsets()
        requested = list(dict.fromkeys(chunk_ids))
        found: dict[str, dict[str, Any]] = {}

        with self.corpus_path.open("rb") as source:
            for chunk_id in requested:
                location = offsets.get(chunk_id)
                if not location:
                    continue
                offset, length = location
                source.seek(offset)
                record = json.loads(source.read(length))
                if record.get("chunk_id") == chunk_id:
                    found[chunk_id] = record

        return found

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        """Return one full chunk record, or None when not found."""
        return self.get_many([chunk_id]).get(chunk_id)
