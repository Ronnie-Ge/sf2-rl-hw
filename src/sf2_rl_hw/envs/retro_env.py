from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple
import shutil
import zipfile

import numpy as np
from PIL import Image

from ..config import EnvironmentConfig, RewardConfig
from ..rewards import RewardTransition, compute_reward
from ..utils.paths import project_root


def _import_gym() -> Any:
    try:
        import gym
    except ImportError:
        import gymnasium as gym
    return gym


_GYM = _import_gym()


class StreetFighterEnvWrapper(_GYM.Wrapper):
    def __init__(
        self,
        env: Any,
        env_config: EnvironmentConfig,
        reward_config: RewardConfig,
        render: bool,
    ) -> None:
        super().__init__(env)
        self.env_config = env_config
        self.reward_config = reward_config
        self.render_enabled = render
        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=env_config.frame_stack)
        self.total_env_steps = 0
        self.prev_agent_hp = reward_config.full_hp
        self.prev_enemy_hp = reward_config.full_hp
        self._reset_returns_info = False
        self._step_returns_truncated = False

        if env_config.grayscale:
            obs_shape = (env_config.height, env_config.width, env_config.frame_stack)
        else:
            obs_shape = (env_config.height, env_config.width, 3)

        self.observation_space = _GYM.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, **kwargs: Any) -> Any:
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
            self._reset_returns_info = True
        else:
            observation, info = reset_result, {}

        processed = self._process_frame(observation)
        self.frame_buffer.clear()
        for _ in range(self.env_config.frame_stack):
            self.frame_buffer.append(processed)

        self.total_env_steps = 0
        self.prev_agent_hp = int(info.get("agent_hp", self.reward_config.full_hp))
        self.prev_enemy_hp = int(info.get("enemy_hp", self.reward_config.full_hp))

        normalized_info = self._normalize_info(info, observation, False)
        stacked = self._stack_observation()
        if self._reset_returns_info:
            return stacked, normalized_info
        return stacked

    def step(self, action: Any) -> Any:
        done = False
        raw_done = False
        latest_observation = None
        latest_info: Dict[str, Any] = {}

        for _ in range(self.env_config.frame_skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                observation, _, terminated, truncated, info = step_result
                self._step_returns_truncated = True
                raw_done = bool(terminated or truncated)
            else:
                observation, _, raw_done, info = step_result

            latest_observation = observation
            latest_info = dict(info)
            self.frame_buffer.append(self._process_frame(observation))
            self.total_env_steps += 1

            if self.render_enabled:
                self.env.render()

            if raw_done:
                break

        if latest_observation is None:
            raise RuntimeError("Environment step did not produce an observation.")

        normalized_info = self._normalize_info(latest_info, latest_observation, raw_done)
        transition = RewardTransition(
            prev_agent_hp=self.prev_agent_hp,
            prev_enemy_hp=self.prev_enemy_hp,
            agent_hp=normalized_info["agent_hp"],
            enemy_hp=normalized_info["enemy_hp"],
            env_step=normalized_info["env_step"],
            result=normalized_info["result"],
        )
        reward, breakdown = compute_reward(transition, self.reward_config)
        normalized_info["reward_breakdown"] = breakdown.to_dict()
        self.prev_agent_hp = normalized_info["agent_hp"]
        self.prev_enemy_hp = normalized_info["enemy_hp"]

        done = normalized_info["round_done"]
        if not self.env_config.reset_round:
            done = False

        stacked = self._stack_observation()
        if self._step_returns_truncated:
            return stacked, reward, done, False, normalized_info
        return stacked, reward, done, normalized_info

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
            return
        try:
            self.env.reset(seed=seed)
        except TypeError:
            return

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(frame)
        if self.env_config.grayscale:
            image = image.convert("L")
        image = image.resize((self.env_config.width, self.env_config.height), Image.BILINEAR)

        processed = np.asarray(image, dtype=np.uint8)
        if self.env_config.grayscale:
            return processed
        if processed.ndim == 2:
            return np.repeat(processed[:, :, None], 3, axis=2)
        return processed

    def _stack_observation(self) -> np.ndarray:
        if self.env_config.grayscale:
            return np.stack(list(self.frame_buffer), axis=-1)

        group_size = self.env_config.frame_stack // 3
        channels = []
        for channel_index in range(3):
            source_index = channel_index * group_size + (group_size - 1)
            channels.append(self.frame_buffer[source_index][:, :, channel_index])
        return np.stack(channels, axis=-1)

    def _normalize_info(
        self,
        info: Dict[str, Any],
        raw_frame: np.ndarray,
        raw_done: bool,
    ) -> Dict[str, Any]:
        normalized = dict(info)
        agent_hp = _coerce_hp(info, ("agent_hp", "health", "player_health"), self.prev_agent_hp)
        enemy_hp = _coerce_hp(info, ("enemy_hp", "enemy_health", "opponent_health"), self.prev_enemy_hp)
        result = _determine_result(agent_hp, enemy_hp, raw_done)

        normalized["agent_hp"] = agent_hp
        normalized["enemy_hp"] = enemy_hp
        normalized["round_done"] = result != "ongoing" or raw_done
        normalized["result"] = result
        normalized["env_step"] = self.total_env_steps
        normalized["frame"] = np.asarray(raw_frame, dtype=np.uint8)
        return normalized


def _coerce_hp(info: Dict[str, Any], candidates: Tuple[str, ...], default: int) -> int:
    for key in candidates:
        if key in info:
            return int(info[key])
    return int(default)


def _determine_result(agent_hp: int, enemy_hp: int, raw_done: bool) -> str:
    if enemy_hp < 0 and agent_hp < 0:
        return "draw"
    if enemy_hp < 0:
        return "win"
    if agent_hp < 0:
        return "lose"
    if raw_done:
        return "done"
    return "ongoing"


def build_retro_env(
    env_config: EnvironmentConfig,
    reward_config: RewardConfig,
    seed: Optional[int] = None,
    render: Optional[bool] = None,
    monitor: bool = False,
) -> Any:
    try:
        import retro  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gym-retro is not installed. Install the runtime dependencies before building the env."
        ) from exc

    register_retro_integration(retro)

    retro_kwargs: Dict[str, Any] = {
        "game": env_config.game,
        "state": env_config.state,
        "use_restricted_actions": retro.Actions.FILTERED,
        "obs_type": retro.Observations.IMAGE,
    }
    if hasattr(retro, "data") and hasattr(retro.data, "Integrations"):
        custom_only = getattr(retro.data.Integrations, "CUSTOM_ONLY", None)
        if custom_only is not None:
            retro_kwargs["inttype"] = custom_only
    if env_config.scenario:
        retro_kwargs["scenario"] = env_config.scenario

    env = retro.make(**retro_kwargs)
    wrapped = StreetFighterEnvWrapper(
        env=env,
        env_config=env_config,
        reward_config=reward_config,
        render=env_config.render if render is None else render,
    )
    if seed is not None:
        wrapped.seed(seed)

    return wrapped


def make_env_factory(
    env_config: EnvironmentConfig,
    reward_config: RewardConfig,
    seed: int,
    render: Optional[bool] = None,
    monitor: bool = False,
) -> Callable[[], Any]:
    def _factory() -> Any:
        return build_retro_env(
            env_config=env_config,
            reward_config=reward_config,
            seed=seed,
            render=render,
            monitor=monitor,
        )

    return _factory


def prepare_integration_assets(env_config: EnvironmentConfig) -> Path:
    rom_path = Path(env_config.rom_path)
    if not rom_path.exists():
        raise FileNotFoundError(f"Configured ROM path does not exist: {rom_path}")

    game_dir = integration_game_dir(env_config.game)
    game_dir.mkdir(parents=True, exist_ok=True)

    rom_target = game_dir / "rom.md"
    if rom_path.suffix.lower() == ".zip":
        _extract_rom_archive(rom_path, rom_target)
    elif not rom_target.exists():
        try:
            rom_target.symlink_to(rom_path)
        except OSError:
            shutil.copy2(rom_path, rom_target)

    return game_dir


def register_retro_integration(retro_module: Any) -> None:
    custom_root = integration_root()
    if not custom_root.exists():
        return

    data_module = getattr(retro_module, "data", None)
    integrations = getattr(data_module, "Integrations", None)
    if integrations is None:
        return

    add_custom_path = getattr(integrations, "add_custom_path", None)
    if callable(add_custom_path):
        add_custom_path(str(custom_root))


def integration_root() -> Path:
    return project_root() / "retro_data"


def integration_game_dir(game: str) -> Path:
    return integration_root() / game


def _extract_rom_archive(archive_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not members:
            raise RuntimeError(f"No ROM file found in archive: {archive_path}")
        member = members[0]
        with archive.open(member) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)


def build_vector_env(
    env_config: EnvironmentConfig,
    reward_config: RewardConfig,
    num_envs: int,
    seed: int,
) -> Any:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        make_env_factory(
            env_config=env_config,
            reward_config=reward_config,
            seed=seed + index,
            render=False,
            monitor=False,
        )
        for index in range(num_envs)
    ]
    return SubprocVecEnv(factories)
