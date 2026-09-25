"""Category inference, jurisdiction inference, and sidecar metadata parsing.

Priority for both fields:
    sidecar .meta.yaml  >  ingest.categories[name] config default  >  "" (unknown)

Sidecar convention: ``{source_doc}.meta.yaml`` placed alongside the document.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Normalise category/folder names: lowercase, collapse whitespace/hyphens/underscores.
_NORM_RE = re.compile(r"[\s\-]+")


def _normalize(name: str) -> str:
    return _NORM_RE.sub("_", name.strip().lower())


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class DocMeta:
    """Resolved metadata for a single source document.

    Attributes:
        category:     Corpus category string (e.g. ``"acts"``), or ``""`` if unknown.
        jurisdiction: Jurisdiction string (e.g. ``"India"``), or ``""`` if unknown.
        year:         Publication/enactment year from sidecar, or ``None``.
        short_title:  Human-readable title from sidecar, or ``""``.
        sidecar_path: Absolute path to the sidecar file, or ``""`` if absent.
    """

    category: str
    jurisdiction: str
    year: int | None = None
    short_title: str = ""
    sidecar_path: str = ""
    parties: list[str] | None = None
    effective_date: str | None = None
    governing_law: str | None = None


# ---------------------------------------------------------------------------
# Sidecar loading
# ---------------------------------------------------------------------------

def load_sidecar(doc_path: Path) -> dict[str, Any]:
    """Load ``{stem}.meta.yaml`` next to *doc_path*.

    Returns ``{}`` if absent. On malformed YAML, logs a warning and
    returns ``{}`` (never raises).
    """
    sidecar = doc_path.parent / (doc_path.stem + ".meta.yaml")
    if not sidecar.is_file():
        return {}
    try:
        data = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("malformed sidecar %s: %s — ignoring", sidecar, exc)
        return {}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _category_map(categories_cfg: list[dict]) -> dict[str, str]:
    """Map normalized folder/category name → canonical category name."""
    return {_normalize(c["name"]): c["name"] for c in categories_cfg}


def infer_category(
    doc_path: Path,
    categories_cfg: list[dict],
    sidecar: dict[str, Any] | None = None,
) -> str:
    """Infer the corpus category for a document.

    Priority: sidecar ``category`` key > parent folder name matched against
    ``categories_cfg`` names > ``""`` (unknown, logged as warning).
    """
    if sidecar and "category" in sidecar:
        return _normalize(str(sidecar["category"]))

    folder_norm = _normalize(doc_path.parent.name)
    cat_map = _category_map(categories_cfg)
    if folder_norm in cat_map:
        return cat_map[folder_norm]

    logger.warning(
        "document %r is in unrecognized folder %r; category set to ''",
        doc_path.name,
        doc_path.parent.name,
    )
    return ""


def infer_jurisdiction(
    category: str,
    categories_cfg: list[dict],
    sidecar: dict[str, Any] | None = None,
) -> str:
    """Infer jurisdiction string.

    Priority: sidecar ``jurisdiction`` key > ``default_jurisdiction`` from
    the matching category config entry > ``""`` (unknown).
    """
    if sidecar and "jurisdiction" in sidecar:
        return str(sidecar["jurisdiction"])

    for cat_cfg in categories_cfg:
        if cat_cfg["name"] == category:
            return str(cat_cfg.get("default_jurisdiction", "") or "")
    return ""


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_doc_meta(
    doc_path: Path,
    categories_cfg: list[dict],
) -> DocMeta:
    """Fully resolve all metadata for *doc_path* in one call.

    Loads the sidecar (if present), then runs both inference functions.
    """
    sidecar = load_sidecar(doc_path)
    category = infer_category(doc_path, categories_cfg, sidecar)
    jurisdiction = infer_jurisdiction(category, categories_cfg, sidecar)

    sidecar_path = ""
    sidecar_file = doc_path.parent / (doc_path.stem + ".meta.yaml")
    if sidecar_file.is_file():
        sidecar_path = str(sidecar_file)

    parties = sidecar.get("parties")
    if isinstance(parties, list):
        parties = [str(p) for p in parties]
    else:
        parties = None

    return DocMeta(
        category=category,
        jurisdiction=jurisdiction,
        year=sidecar.get("year"),
        short_title=str(sidecar.get("short_title", "") or ""),
        sidecar_path=sidecar_path,
        parties=parties,
        effective_date=sidecar.get("effective_date"),
        governing_law=sidecar.get("governing_law"),
    )
