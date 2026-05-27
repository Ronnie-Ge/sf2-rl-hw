from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import build_ppo_agent, save_agent
from .config import ExperimentConfig, load_experiment_config
from .envs import build_vector_env, prepare_integration_assets
from .evaluate import evaluate_checkpoint
from .utils.logging import dump_json, emit_section
from .utils.paths import (
    RunArtifacts,
    checkpoint_filename,
    prepare_run_artifacts,
)
from .utils.seed import set_global_seed


class ArtifactTriggerCallback:
    def __init__(self, config: ExperimentConfig, artifacts: RunArtifacts) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "ArtifactTriggerCallback") -> None:
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

        self.callback = _Callback(self)
        self.config = config
        self.artifacts = artifacts
        self.next_checkpoint = config.ppo.checkpoint_freq
        self.next_evaluation = config.evaluation.trigger_every_n_timesteps

    def on_step(self, callback: Any) -> bool:
        current_step = int(callback.num_timesteps)
        should_save = current_step >= self.next_checkpoint
        should_eval = current_step >= self.next_evaluation

        if not any((should_save, should_eval)):
            return True

        checkpoint_path = None
        if should_save:
            checkpoint_path = self.artifacts.checkpoints_dir / checkpoint_filename(current_step)
            save_agent(callback.model, checkpoint_path)

            while self.next_checkpoint <= current_step:
                self.next_checkpoint += self.config.ppo.checkpoint_freq

        if should_eval:
            evaluate_checkpoint(
                config=self.config,
                checkpoint_path=checkpoint_path,
                output_dir=self.artifacts.eval_dir,
                model=callback.model,
                checkpoint_step=current_step,
            )
            while self.next_evaluation <= current_step:
                self.next_evaluation += self.config.evaluation.trigger_every_n_timesteps

        return True


def run_training(config_path: Path) -> None:
    config, resolved = load_experiment_config(config_path)
    artifacts = prepare_run_artifacts(config.runtime.output_dir, config.experiment_name)
    dump_json(artifacts.run_dir / "resolved_config.json", resolved)
    dump_json(
        artifacts.run_dir / "run_metadata.json",
        {
            "experiment_name": config.experiment_name,
            "notes": config.notes,
            "tags": config.tags,
            "run_name": artifacts.run_name,
            "config_path": str(config_path.resolve()),
        },
    )

    emit_section(
        "Training",
        [
            f"run_dir={artifacts.run_dir}",
            f"checkpoints_dir={artifacts.checkpoints_dir}",
            f"eval_dir={artifacts.eval_dir}",
            f"timesteps={config.ppo.total_timesteps}",
            f"num_envs={config.ppo.num_envs}",
        ],
    )

    set_global_seed(config.runtime.seed)
    prepare_integration_assets(config.env)
    env = build_vector_env(
        env_config=config.env,
        reward_config=config.reward,
        num_envs=config.ppo.num_envs,
        seed=config.runtime.seed,
    )
    try:
        model = build_ppo_agent(
            config=config.ppo,
            env=env,
            device=config.runtime.device,
            tensorboard_log=artifacts.logs_dir,
        )
        callback = ArtifactTriggerCallback(config, artifacts).callback
        model.learn(total_timesteps=int(config.ppo.total_timesteps), callback=callback)

        final_step = int(model.num_timesteps)
        final_checkpoint = artifacts.checkpoints_dir / checkpoint_filename(final_step)
        save_agent(model, final_checkpoint)
        evaluate_checkpoint(
            config=config,
            checkpoint_path=final_checkpoint,
            output_dir=artifacts.eval_dir,
            model=model,
            checkpoint_step=final_step,
        )
    finally:
        env.close()
