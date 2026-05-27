from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..config import PPOConfig


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def scheduler(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return scheduler


def build_ppo_agent(
    config: PPOConfig,
    env: Any,
    device: str,
    tensorboard_log: Optional[Path] = None,
) -> Any:
    from stable_baselines3 import PPO

    return PPO(
        config.policy,
        env,
        device=device,
        verbose=config.verbose,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        learning_rate=linear_schedule(config.learning_rate_start, config.learning_rate_end),
        clip_range=linear_schedule(config.clip_range_start, config.clip_range_end),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )


def load_agent(
    checkpoint_path: Path,
    env: Optional[Any],
    config: PPOConfig,
    device: str,
) -> Any:
    from stable_baselines3 import PPO

    custom_objects = {
        "learning_rate": linear_schedule(config.learning_rate_start, config.learning_rate_end),
        "clip_range": linear_schedule(config.clip_range_start, config.clip_range_end),
        "n_steps": config.n_steps,
    }
    return PPO.load(
        str(checkpoint_path),
        env=env,
        device=device,
        custom_objects=custom_objects,
    )


def save_agent(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def predict_action(model: Any, observation: Any, deterministic: bool) -> Any:
    action, _ = model.predict(observation, deterministic=deterministic)
    return action
