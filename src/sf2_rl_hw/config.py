from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

DEFAULT_BUTTONS = [
    "B",
    "A",
    "MODE",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "C",
    "Y",
    "X",
    "Z",
]


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts"
    experiment_name: str = ""


@dataclass
class EnvironmentConfig:
    game: str = "StreetFighterIISpecialChampionEdition-Genesis"
    state: str = "Champion.Level12.RyuVsBison"
    scenario: str = "default"
    rom_path: str = ""
    buttons: List[str] = field(default_factory=lambda: list(DEFAULT_BUTTONS))
    frame_skip: int = 6
    frame_stack: int = 9
    width: int = 128
    height: int = 100
    grayscale: bool = False
    render: bool = False
    headless: bool = True
    reset_round: bool = True


@dataclass
class RewardConfig:
    profile: str = "reference_v1"
    damage_dealt_weight: float = 3.0
    damage_taken_weight: float = 1.0
    win_bonus: float = 176.0
    lose_penalty: float = 176.0
    time_penalty: float = 0.0
    normalize_factor: float = 0.001
    full_hp: int = 176


@dataclass
class PPOConfig:
    policy: str = "CnnPolicy"
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    learning_rate_start: float = 2.5e-4
    learning_rate_end: float = 2.5e-6
    clip_range_start: float = 0.15
    clip_range_end: float = 0.025
    batch_size: int = 512
    n_steps: int = 512
    n_epochs: int = 4
    gamma: float = 0.94
    ent_coef: float = 0.0
    verbose: int = 1
    checkpoint_freq: int = 500_000


@dataclass
class EvaluationConfig:
    checkpoint_path: str = ""
    episodes: int = 5
    deterministic: bool = True
    render: bool = False
    save_metrics: bool = True
    trigger_every_n_timesteps: int = 500_000


@dataclass
class RecordingConfig:
    enabled: bool = True
    checkpoint_path: str = ""
    episodes: int = 1
    deterministic: bool = True
    fps: int = 60
    overlay: bool = True
    save_video: bool = True
    render: bool = False


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    @property
    def experiment_name(self) -> str:
        return self.runtime.experiment_name or self.name

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["runtime"]["experiment_name"] = self.experiment_name
        return payload


def load_experiment_config(path: Path) -> Tuple[ExperimentConfig, Dict[str, Any]]:
    resolved_path = path.resolve()
    raw = _load_yaml_with_extends(resolved_path)
    raw = _apply_env_overrides(raw)
    raw = _resolve_path_fields(raw, resolved_path.parent)

    runtime_raw = raw.get("runtime", {})
    if not runtime_raw.get("experiment_name"):
        runtime_raw["experiment_name"] = raw.get("name", "baseline")

    cfg = ExperimentConfig(
        name=raw.get("name", "baseline"),
        notes=raw.get("notes", ""),
        tags=raw.get("tags", []),
        runtime=RuntimeConfig(**runtime_raw),
        env=EnvironmentConfig(**raw.get("env", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        recording=RecordingConfig(**raw.get("recording", {})),
    )
    _validate_config(cfg, resolved_path)
    return cfg, cfg.to_dict()


def _load_yaml_with_extends(path: Path) -> Dict[str, Any]:
    raw = _read_yaml(path)
    base_ref = raw.pop("extends", None)
    if not base_ref:
        return raw

    base_path = (path.parent / base_ref).resolve()
    base_config = _load_yaml_with_extends(base_path)
    return _deep_merge(base_config, raw)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Run `uv sync` first.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    overridden = _deep_merge({}, raw)
    env_cfg = overridden.setdefault("env", {})
    eval_cfg = overridden.setdefault("evaluation", {})
    record_cfg = overridden.setdefault("recording", {})

    rom_override = os.getenv("SF2_ROM_PATH")
    if rom_override:
        env_cfg["rom_path"] = rom_override

    eval_checkpoint = os.getenv("SF2_EVAL_CHECKPOINT")
    if eval_checkpoint:
        eval_cfg["checkpoint_path"] = eval_checkpoint

    record_checkpoint = os.getenv("SF2_RECORD_CHECKPOINT")
    if record_checkpoint:
        record_cfg["checkpoint_path"] = record_checkpoint

    return overridden


def _resolve_path_fields(raw: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    resolved = _deep_merge({}, raw)
    runtime_cfg = resolved.setdefault("runtime", {})
    env_cfg = resolved.setdefault("env", {})
    eval_cfg = resolved.setdefault("evaluation", {})
    record_cfg = resolved.setdefault("recording", {})

    project_root = _find_project_root(config_dir)
    runtime_cfg["output_dir"] = str(_resolve_path(runtime_cfg.get("output_dir", "artifacts"), project_root))
    env_cfg["rom_path"] = _resolve_optional_path(env_cfg.get("rom_path", ""), project_root)
    eval_cfg["checkpoint_path"] = _resolve_optional_path(eval_cfg.get("checkpoint_path", ""), project_root)
    record_cfg["checkpoint_path"] = _resolve_optional_path(record_cfg.get("checkpoint_path", ""), project_root)
    return resolved


def _resolve_path(value: str, base_dir: Path, allow_empty: bool = False) -> Path:
    if not value:
        if allow_empty:
            return Path()
        raise ValueError("Path value cannot be empty.")

    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_path(value: str, base_dir: Path) -> str:
    if not value:
        return ""
    return str(_resolve_path(value, base_dir))


def _find_project_root(start_dir: Path) -> Path:
    current = start_dir.resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start_dir.resolve()


def _validate_config(config: ExperimentConfig, config_path: Path) -> None:
    if not config.experiment_name:
        raise ValueError(f"runtime.experiment_name cannot be empty: {config_path}")
    if config.env.frame_skip <= 0:
        raise ValueError("env.frame_skip must be greater than 0")
    if config.env.frame_stack <= 0:
        raise ValueError("env.frame_stack must be greater than 0")
    if config.env.width <= 0 or config.env.height <= 0:
        raise ValueError("env.width and env.height must be greater than 0")
    if not config.env.buttons:
        raise ValueError("env.buttons must contain at least one button")
    if len(config.env.buttons) != len(set(config.env.buttons)):
        raise ValueError("env.buttons cannot contain duplicate button names")
    invalid_buttons = [button for button in config.env.buttons if button not in DEFAULT_BUTTONS]
    if invalid_buttons:
        raise ValueError(
            f"env.buttons contains unsupported button names: {invalid_buttons}. "
            f"Supported buttons: {DEFAULT_BUTTONS}"
        )
    if config.ppo.total_timesteps <= 0:
        raise ValueError("ppo.total_timesteps must be greater than 0")
    if config.ppo.num_envs <= 0:
        raise ValueError("ppo.num_envs must be greater than 0")
    if config.ppo.checkpoint_freq <= 0:
        raise ValueError("ppo.checkpoint_freq must be greater than 0")
    if config.evaluation.episodes <= 0:
        raise ValueError("evaluation.episodes must be greater than 0")
    if config.recording.episodes <= 0:
        raise ValueError("recording.episodes must be greater than 0")
    if config.recording.fps <= 0:
        raise ValueError("recording.fps must be greater than 0")
    if config.reward.profile not in {"baseline", "reference_v1"}:
        raise ValueError("reward.profile must be one of: baseline, reference_v1")
    if not config.env.grayscale and config.env.frame_stack % 3 != 0:
        raise ValueError("env.frame_stack must be divisible by 3 when using RGB stacking")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged
