from __future__ import annotations

import math
from typing import Tuple

from ..config import RewardConfig
from .base import RewardBreakdown, RewardTransition


def compute_reward(
    transition: RewardTransition,
    config: RewardConfig,
) -> Tuple[float, RewardBreakdown]:
    if config.profile == "baseline":
        breakdown = compute_baseline_reward(transition, config)
    elif config.profile == "reference_v1":
        breakdown = compute_reference_v1_reward(transition, config)
    else:
        raise ValueError(f"Unsupported reward profile: {config.profile}")

    return breakdown.normalized_reward, breakdown


def compute_baseline_reward(
    transition: RewardTransition,
    config: RewardConfig,
) -> RewardBreakdown:
    damage_dealt = max(0, transition.prev_enemy_hp - transition.enemy_hp)
    damage_taken = max(0, transition.prev_agent_hp - transition.agent_hp)

    breakdown = RewardBreakdown(
        profile="baseline",
        damage_reward=config.damage_dealt_weight * damage_dealt,
        damage_penalty=config.damage_taken_weight * damage_taken,
        time_penalty=config.time_penalty,
    )

    if transition.result == "win":
        breakdown.win_bonus = config.win_bonus
    elif transition.result == "lose":
        breakdown.lose_penalty = config.lose_penalty

    breakdown.raw_reward = (
        breakdown.damage_reward
        - breakdown.damage_penalty
        + breakdown.win_bonus
        - breakdown.lose_penalty
        - breakdown.time_penalty
    )
    breakdown.normalized_reward = breakdown.raw_reward * config.normalize_factor
    return breakdown


def compute_reference_v1_reward(
    transition: RewardTransition,
    config: RewardConfig,
) -> RewardBreakdown:
    damage_dealt = max(0, transition.prev_enemy_hp - transition.enemy_hp)
    damage_taken = max(0, transition.prev_agent_hp - transition.agent_hp)

    breakdown = RewardBreakdown(
        profile="reference_v1",
        damage_reward=config.damage_dealt_weight * damage_dealt,
        damage_penalty=config.damage_taken_weight * damage_taken,
        time_penalty=config.time_penalty,
    )

    if transition.result == "lose":
        breakdown.lose_penalty = config.lose_penalty * math.pow(
            config.full_hp,
            (transition.enemy_hp + 1) / (config.full_hp + 1),
        )
        breakdown.raw_reward = -breakdown.lose_penalty
    elif transition.result == "win":
        breakdown.win_bonus = config.win_bonus * math.pow(
            config.full_hp,
            (transition.agent_hp + 1) / (config.full_hp + 1),
        )
        breakdown.raw_reward = breakdown.win_bonus
    else:
        breakdown.raw_reward = (
            breakdown.damage_reward
            - breakdown.damage_penalty
            - breakdown.time_penalty
        )

    breakdown.normalized_reward = breakdown.raw_reward * config.normalize_factor
    return breakdown
