from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .agents import predict_action


@dataclass
class EpisodeMetrics:
    episode_index: int
    episode_return: float
    result: str
    final_agent_hp: int
    final_enemy_hp: int
    final_hp_diff: int
    episode_length: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_policy_episodes(
    model: Any,
    env: Any,
    episodes: int,
    deterministic: bool,
    render: bool,
    experiment_name: str,
    checkpoint_step: Optional[int],
    max_post_result_steps: int = 0,
    episode_start_callback: Optional[Callable[[int], None]] = None,
    step_callback: Optional[Callable[[int, np.ndarray, Dict[str, Any]], None]] = None,
    episode_end_callback: Optional[Callable[[int, EpisodeMetrics], None]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    episode_summaries: List[EpisodeMetrics] = []
    action_labels = action_labels_from_env(env)

    for episode_index in range(episodes):
        if episode_start_callback:
            episode_start_callback(episode_index)

        observation, reset_info = reset_env(env)
        done = False
        episode_return = 0.0
        episode_length = 0
        latest_info = dict(reset_info)
        post_result_steps_remaining: Optional[int] = None

        while not done:
            action = predict_action(model, observation, deterministic=deterministic)
            observation, reward, done, info = step_env(env, action)
            latest_info = dict(info)
            episode_return += float(reward)
            episode_length += 1

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            overlay_state = {
                "experiment_name": experiment_name,
                "checkpoint_step": checkpoint_step if checkpoint_step is not None else "manual",
                "episode": episode_index + 1,
                "env_step": info.get("env_step", episode_length),
                "action": action_to_text(action, action_labels),
                "instant_reward": float(reward),
                "episode_return": episode_return,
                "agent_hp": int(info.get("agent_hp", 0)),
                "enemy_hp": int(info.get("enemy_hp", 0)),
                "result": info.get("result", "ongoing"),
            }
            frame = info.get("frame")
            if frame is None:
                frame = observation_to_frame(observation)
            if step_callback:
                step_callback(episode_index, np.asarray(frame, dtype=np.uint8), overlay_state)

            if not done and max_post_result_steps > 0 and info.get("result", "ongoing") != "ongoing":
                if post_result_steps_remaining is None:
                    post_result_steps_remaining = max_post_result_steps
                else:
                    post_result_steps_remaining -= 1
                    if post_result_steps_remaining <= 0:
                        done = True

        summary = EpisodeMetrics(
            episode_index=episode_index + 1,
            episode_return=episode_return,
            result=str(latest_info.get("result", "done")),
            final_agent_hp=int(latest_info.get("agent_hp", 0)),
            final_enemy_hp=int(latest_info.get("enemy_hp", 0)),
            final_hp_diff=int(latest_info.get("agent_hp", 0)) - int(latest_info.get("enemy_hp", 0)),
            episode_length=episode_length,
        )
        episode_summaries.append(summary)
        if episode_end_callback:
            episode_end_callback(episode_index, summary)

    summary_payload = summarize_episodes(
        experiment_name=experiment_name,
        checkpoint_step=checkpoint_step,
        episodes=episode_summaries,
    )
    return summary_payload, [episode.to_dict() for episode in episode_summaries]


def summarize_episodes(
    experiment_name: str,
    checkpoint_step: Optional[int],
    episodes: List[EpisodeMetrics],
) -> Dict[str, Any]:
    if not episodes:
        raise ValueError("At least one episode is required to summarize metrics.")

    total_episodes = len(episodes)
    win_count = sum(1 for episode in episodes if episode.result == "win")
    returns = [episode.episode_return for episode in episodes]
    hp_diffs = [episode.final_hp_diff for episode in episodes]
    lengths = [episode.episode_length for episode in episodes]

    return {
        "experiment_name": experiment_name,
        "checkpoint_step": checkpoint_step,
        "episodes": total_episodes,
        "win_rate": win_count / total_episodes,
        "mean_episode_return": float(np.mean(returns)),
        "mean_final_hp_diff": float(np.mean(hp_diffs)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def reset_env(env: Any) -> Tuple[Any, Dict[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, dict(info)
    return result, {}


def step_env(env: Any, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), dict(info)
    observation, reward, done, info = result
    return observation, float(reward), bool(done), dict(info)


def action_to_text(action: Any, labels: Optional[List[str]] = None) -> str:
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return str(action.item())
        if labels and action.ndim == 1 and len(labels) == action.shape[0]:
            pressed = [label for label, value in zip(labels, action.tolist()) if float(value) >= 0.5]
            return "+".join(pressed) if pressed else "(none)"
        return "[" + ",".join(str(item) for item in action.flatten().tolist()) + "]"
    return str(action)


def action_labels_from_env(env: Any) -> Optional[List[str]]:
    visited = set()
    current = env
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        buttons = getattr(current, "buttons", None)
        if buttons:
            return [str(button) for button in buttons]
        current = getattr(current, "env", None)

    unwrapped = getattr(env, "unwrapped", None)
    buttons = getattr(unwrapped, "buttons", None)
    if buttons:
        return [str(button) for button in buttons]
    return None


def observation_to_frame(observation: Any) -> np.ndarray:
    array = np.asarray(observation, dtype=np.uint8)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 1:
        return np.repeat(array, 3, axis=2)
    if array.ndim == 3 and array.shape[-1] >= 3:
        return array[:, :, :3]
    raise ValueError("Cannot convert observation to RGB frame.")
