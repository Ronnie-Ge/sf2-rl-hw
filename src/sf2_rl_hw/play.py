from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agents import load_agent
from .config import load_experiment_config
from .envs import build_retro_env, prepare_integration_assets
from .record import record_checkpoint
from .rollout import run_policy_episodes
from .utils.logging import dump_json, emit_section
from .utils.paths import prepare_run_artifacts, resolve_checkpoint_path


def run_play(config_path: Path, checkpoint_override: Optional[Path] = None) -> None:
    config, resolved = load_experiment_config(config_path)
    checkpoint_path = resolve_checkpoint_path(
        explicit_checkpoint=checkpoint_override,
        configured_checkpoint=config.play.checkpoint_path,
        output_dir=config.runtime.output_dir,
        experiment_name=config.experiment_name,
    )
    artifacts = prepare_run_artifacts(config.runtime.output_dir, config.experiment_name)
    dump_json(artifacts.run_dir / "resolved_config.json", resolved)

    if config.play.save_video:
        payload = record_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=artifacts.videos_dir,
            episodes_override=config.play.episodes,
            overlay_override=config.play.overlay,
            render_override=config.play.render,
            save_video_override=True,
            deterministic_override=config.play.deterministic,
        )
        if payload["errors"]:
            raise RuntimeError(f"Play recording failed: {payload['errors']}")
        emit_section(
            "Play",
            [
                f"mode=render+record",
                f"checkpoint={checkpoint_path}",
                f"videos={len(payload['videos'])}",
            ],
        )
        return

    prepare_integration_assets(config.env)
    env = build_retro_env(
        env_config=config.env,
        reward_config=config.reward,
        seed=config.runtime.seed,
        render=config.play.render,
        monitor=False,
    )
    try:
        model = load_agent(
            checkpoint_path=checkpoint_path,
            env=env,
            config=config.ppo,
            device=config.runtime.device,
        )
        summary, _ = run_policy_episodes(
            model=model,
            env=env,
            episodes=config.play.episodes,
            deterministic=config.play.deterministic,
            render=config.play.render,
            experiment_name=config.experiment_name,
            checkpoint_step=None,
        )
    finally:
        env.close()

    emit_section(
        "Play",
        [
            f"checkpoint={checkpoint_path}",
            f"episodes={config.play.episodes}",
            f"win_rate={summary['win_rate']:.2%}",
            f"mean_episode_return={summary['mean_episode_return']:.4f}",
        ],
    )
