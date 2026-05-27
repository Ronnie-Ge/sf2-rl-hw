from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .agents import load_agent
from .config import ExperimentConfig, load_experiment_config
from .envs import build_retro_env, prepare_integration_assets
from .rollout import run_policy_episodes
from .utils.logging import dump_json, emit_section
from .utils.paths import (
    checkpoint_step_from_path,
    prepare_run_artifacts,
    resolve_checkpoint_path,
)


def run_evaluation(config_path: Path, checkpoint_override: Optional[Path] = None) -> None:
    config, resolved = load_experiment_config(config_path)
    checkpoint_path = resolve_checkpoint_path(
        explicit_checkpoint=checkpoint_override,
        configured_checkpoint=config.evaluation.checkpoint_path,
        output_dir=config.runtime.output_dir,
        experiment_name=config.experiment_name,
    )
    artifacts = prepare_run_artifacts(config.runtime.output_dir, config.experiment_name)
    dump_json(artifacts.run_dir / "resolved_config.json", resolved)

    summary, episodes = evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=artifacts.eval_dir,
    )
    emit_section(
        "Evaluation",
        [
            f"run_dir={artifacts.run_dir}",
            f"checkpoint={checkpoint_path}",
            f"win_rate={summary['win_rate']:.2%}",
            f"mean_episode_return={summary['mean_episode_return']:.4f}",
            f"mean_final_hp_diff={summary['mean_final_hp_diff']:.2f}",
        ],
    )
    if not episodes:
        print("No evaluation episodes were recorded.")


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    model: Optional[Any] = None,
    checkpoint_step: Optional[int] = None,
) -> Tuple[Dict[str, Any], Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_step is None:
        if checkpoint_path is None:
            step_value = None
        else:
            step_value = checkpoint_step_from_path(checkpoint_path)
    else:
        step_value = checkpoint_step
    step_dir = output_dir / f"step_{step_value if step_value is not None else 'manual'}"
    step_dir.mkdir(parents=True, exist_ok=True)
    prepare_integration_assets(config.env)

    env = build_retro_env(
        env_config=config.env,
        reward_config=config.reward,
        seed=config.runtime.seed,
        render=config.evaluation.render,
        monitor=False,
    )
    try:
        if model is not None:
            eval_model = model
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when no model instance is provided")
            eval_model = load_agent(
                checkpoint_path=checkpoint_path,
                env=env,
                config=config.ppo,
                device=config.runtime.device,
            )
        summary, episodes = run_policy_episodes(
            model=eval_model,
            env=env,
            episodes=config.evaluation.episodes,
            deterministic=config.evaluation.deterministic,
            render=config.evaluation.render,
            experiment_name=config.experiment_name,
            checkpoint_step=step_value,
        )
        if config.evaluation.save_metrics:
            dump_json(step_dir / "metrics.json", summary)
            dump_json(step_dir / "episode_metrics.json", episodes)
        return summary, episodes
    finally:
        env.close()
