"""Raw API snapshots, saved as data/raw/<source>/<name>.json so a run can be repeated offline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _path(raw_dir: str | Path, source: str, name: str) -> Path:
    for part in (source, name):
        if not _NAME.fullmatch(part):
            raise ValueError(f"unsafe snapshot name: {part!r}")
    return Path(raw_dir) / source / f"{name}.json"


def save_json(raw_dir: str | Path, source: str, name: str, payload: Any) -> Path:
    """Write payload to raw_dir/source/name.json, creating folders. Returns the path."""
    path = _path(raw_dir, source, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_json(raw_dir: str | Path, source: str, name: str) -> Any | None:
    """The saved payload, or None when the snapshot does not exist."""
    path = _path(raw_dir, source, name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_names(raw_dir: str | Path, source: str, prefix: str = "") -> list[str]:
    """Sorted snapshot names in raw_dir/source that start with prefix."""
    folder = Path(raw_dir) / source
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json") if p.stem.startswith(prefix))


def latest_name(raw_dir: str | Path, source: str, prefix: str = "") -> str | None:
    """The last name in sorted order (names embed ISO dates, so this is the newest), or None."""
    names = list_names(raw_dir, source, prefix)
    return names[-1] if names else None
