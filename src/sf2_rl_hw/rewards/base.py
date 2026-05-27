from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class RewardTransition:
    prev_agent_hp: int
    prev_enemy_hp: int
    agent_hp: int
    enemy_hp: int
    env_step: int = 0
    result: str = "ongoing"


@dataclass
class RewardBreakdown:
    profile: str
    damage_reward: float = 0.0
    damage_penalty: float = 0.0
    win_bonus: float = 0.0
    lose_penalty: float = 0.0
    time_penalty: float = 0.0
    raw_reward: float = 0.0
    normalized_reward: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
