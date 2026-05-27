from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import load_agent
from .config import ExperimentConfig, load_experiment_config
from .envs import build_retro_env, prepare_integration_assets
from .rollout import EpisodeMetrics, run_policy_episodes
from .utils.logging import dump_json, emit_section
from .utils.overlay import draw_overlay
from .utils.paths import (
    checkpoint_step_from_path,
    checkpoints_from_glob,
    latest_checkpoint,
    prepare_run_artifacts,
    resolve_checkpoint_path,
    select_latest_checkpoints,
    video_filename,
)
from .utils.video import VideoFrameWriter


def run_recording(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    latest_count: Optional[int] = None,
    glob_pattern: Optional[str] = None,
) -> None:
    config, resolved = load_experiment_config(config_path)
    artifacts = prepare_run_artifacts(config.runtime.output_dir, config.experiment_name)
    dump_json(artifacts.run_dir / "resolved_config.json", resolved)

    checkpoints = resolve_recording_targets(
        config=config,
        checkpoint_override=checkpoint_override,
        latest_count=latest_count,
        glob_pattern=glob_pattern,
    )
    batch_results: List[Dict[str, Any]] = []
    total_videos = 0
    total_errors = 0

    for checkpoint_path in checkpoints:
        payload = record_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=artifacts.videos_dir,
        )
        batch_results.append(
            {
                "checkpoint": str(checkpoint_path),
                "summary": payload["summary"],
                "videos": payload["videos"],
                "errors": payload["errors"],
            }
        )
        total_videos += len(payload["videos"])
        total_errors += len(payload["errors"])

    dump_json(
        artifacts.videos_dir / "batch_record_manifest.json",
        {
            "experiment_name": config.experiment_name,
            "targets": len(checkpoints),
            "videos": total_videos,
            "errors": total_errors,
            "items": batch_results,
        },
    )

    emit_section(
        "Recording",
        [
            f"run_dir={artifacts.run_dir}",
            f"targets={len(checkpoints)}",
            f"videos={total_videos}",
            f"errors={total_errors}",
            f"output_dir={artifacts.videos_dir}",
        ],
    )
    if total_errors:
        raise RuntimeError(f"Recording completed with {total_errors} error(s). Check batch_record_manifest.json.")


def resolve_recording_targets(
    config: ExperimentConfig,
    checkpoint_override: Optional[Path],
    latest_count: Optional[int],
    glob_pattern: Optional[str],
) -> List[Path]:
    mode_count = sum(
        1
        for value in (
            checkpoint_override is not None,
            latest_count is not None,
            bool(glob_pattern),
            bool(config.recording.checkpoint_path),
        )
        if value
    )
    if mode_count > 1:
        raise ValueError("record accepts only one checkpoint selection mode at a time")

    if checkpoint_override is not None:
        return [checkpoint_override.resolve()]
    if latest_count is not None:
        return select_latest_checkpoints(
            output_dir=config.runtime.output_dir,
            experiment_name=config.experiment_name,
            limit=latest_count,
        )
    if glob_pattern:
        return checkpoints_from_glob(glob_pattern)

    if config.recording.checkpoint_path:
        return [
            resolve_checkpoint_path(
                explicit_checkpoint=None,
                configured_checkpoint=config.recording.checkpoint_path,
                output_dir=config.runtime.output_dir,
                experiment_name=config.experiment_name,
            )
        ]
    return [latest_checkpoint(config.runtime.output_dir, config.experiment_name)]


def record_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_dir: Path,
    model: Optional[Any] = None,
    checkpoint_step: Optional[int] = None,
    episodes_override: Optional[int] = None,
    overlay_override: Optional[bool] = None,
    render_override: Optional[bool] = None,
    save_video_override: Optional[bool] = None,
    fps_override: Optional[int] = None,
    deterministic_override: Optional[bool] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_value = checkpoint_step_from_path(checkpoint_path) if checkpoint_step is None else checkpoint_step
    step_dir = output_dir / f"step_{step_value if step_value is not None else 'manual'}"
    step_dir.mkdir(parents=True, exist_ok=True)

    episodes = episodes_override if episodes_override is not None else config.recording.episodes
    overlay = config.recording.overlay if overlay_override is None else overlay_override
    render = config.recording.render if render_override is None else render_override
    save_video = config.recording.save_video if save_video_override is None else save_video_override
    fps = config.recording.fps if fps_override is None else fps_override
    deterministic = (
        config.recording.deterministic if deterministic_override is None else deterministic_override
    )

    prepare_integration_assets(config.env)
    env = build_retro_env(
        env_config=config.env,
        reward_config=config.reward,
        seed=config.runtime.seed,
        render=render,
        monitor=False,
    )
    current_writer: Dict[str, Any] = {"writer": None, "output_path": None}
    video_outputs: List[str] = []
    errors: List[str] = []

    def on_episode_start(episode_index: int) -> None:
        if not save_video:
            current_writer["writer"] = None
            current_writer["output_path"] = None
            return

        output_path = step_dir / video_filename(
            step=step_value if step_value is not None else 0,
            episode_index=episode_index + 1,
        )
        current_writer["writer"] = VideoFrameWriter(output_path.parent, fps)
        current_writer["output_path"] = output_path

    def on_step(_: int, frame: Any, overlay_state: Dict[str, Any]) -> None:
        writer = current_writer["writer"]
        if writer is None:
            return
        video_frame = draw_overlay(frame, overlay_state) if overlay else frame
        writer.add_frame(video_frame)

    def on_episode_end(_: int, summary: EpisodeMetrics) -> None:
        writer = current_writer["writer"]
        output_path = current_writer["output_path"]
        if writer is None or output_path is None:
            return
        try:
            writer.finalize(output_path)
            video_outputs.append(str(output_path))
        except Exception as exc:
            errors.append(f"episode_{summary.episode_index}: {exc}")

    try:
        record_model = model or load_agent(
            checkpoint_path=checkpoint_path,
            env=env,
            config=config.ppo,
            device=config.runtime.device,
        )
        summary, episodes_payload = run_policy_episodes(
            model=record_model,
            env=env,
            episodes=episodes,
            deterministic=deterministic,
            render=render,
            experiment_name=config.experiment_name,
            checkpoint_step=step_value,
            episode_start_callback=on_episode_start,
            step_callback=on_step,
            episode_end_callback=on_episode_end,
        )
    finally:
        env.close()

    payload = {
        "summary": summary,
        "episodes": episodes_payload,
        "videos": video_outputs,
        "errors": errors,
    }
    dump_json(step_dir / "record_metrics.json", payload)
    return payload
