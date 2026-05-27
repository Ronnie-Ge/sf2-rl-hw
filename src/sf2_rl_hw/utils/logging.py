from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def emit_section(title: str, lines: Iterable[str]) -> None:
    print(f"[{title}]")
    for line in lines:
        print(f"- {line}")


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_metrics(path: Path, summary: Dict[str, Any], episodes: List[Dict[str, Any]]) -> None:
    dump_json(path.parent / "metrics.json", summary)
    dump_json(path.parent / "episode_metrics.json", episodes)
