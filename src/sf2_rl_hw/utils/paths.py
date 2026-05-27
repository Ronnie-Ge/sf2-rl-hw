from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
import re
import glob


@dataclass
class RunArtifacts:
    root_dir: Path
    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    videos_dir: Path
    logs_dir: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_output_root(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return (project_root() / output_path).resolve()


def build_run_name(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{experiment_name}-{timestamp}"


def prepare_run_artifacts(output_dir: str, experiment_name: str) -> RunArtifacts:
    root_dir = resolve_output_root(output_dir)
    run_name = build_run_name(experiment_name)
    run_dir = root_dir / "runs" / experiment_name / run_name
    checkpoints_dir = root_dir / "checkpoints" / experiment_name / run_name
    eval_dir = root_dir / "eval" / experiment_name / run_name
    videos_dir = root_dir / "videos" / experiment_name / run_name
    logs_dir = root_dir / "logs" / experiment_name / run_name

    for path in (run_dir, checkpoints_dir, eval_dir, videos_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        root_dir=root_dir,
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        eval_dir=eval_dir,
        videos_dir=videos_dir,
        logs_dir=logs_dir,
    )


def checkpoint_filename(step: int) -> str:
    return f"ppo_step_{step}.zip"


def video_filename(step: int, episode_index: int) -> str:
    return f"step_{step}_ep_{episode_index:02d}.mp4"


def checkpoint_step_from_path(path: Path) -> Optional[int]:
    match = re.search(r"step_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def latest_checkpoint(output_dir: str, experiment_name: str) -> Path:
    candidates = list_checkpoints(output_dir, experiment_name)
    if not candidates:
        checkpoint_root = resolve_output_root(output_dir) / "checkpoints" / experiment_name
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_root}")
    return candidates[-1]


def list_checkpoints(output_dir: str, experiment_name: str) -> List[Path]:
    root_dir = resolve_output_root(output_dir)
    checkpoint_root = root_dir / "checkpoints" / experiment_name
    candidates = list(checkpoint_root.rglob("*.zip"))
    return sort_checkpoints(candidates)


def select_latest_checkpoints(output_dir: str, experiment_name: str, limit: int) -> List[Path]:
    if limit <= 0:
        raise ValueError("latest checkpoint count must be greater than 0")
    candidates = list_checkpoints(output_dir, experiment_name)
    if not candidates:
        checkpoint_root = resolve_output_root(output_dir) / "checkpoints" / experiment_name
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_root}")
    return candidates[-limit:]


def checkpoints_from_glob(pattern: str) -> List[Path]:
    matches = [Path(match).resolve() for match in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(f"No checkpoints matched glob: {pattern}")
    return sort_checkpoints(matches)


def sort_checkpoints(paths: Iterable[Path]) -> List[Path]:
    return sorted(
        paths,
        key=lambda path: (
            checkpoint_step_from_path(path) is None,
            checkpoint_step_from_path(path) or -1,
            str(path),
        ),
    )


def resolve_checkpoint_path(
    explicit_checkpoint: Optional[Path],
    configured_checkpoint: str,
    output_dir: str,
    experiment_name: str,
) -> Path:
    if explicit_checkpoint:
        return explicit_checkpoint.resolve()
    if configured_checkpoint:
        return Path(configured_checkpoint).resolve()
    return latest_checkpoint(output_dir, experiment_name)
