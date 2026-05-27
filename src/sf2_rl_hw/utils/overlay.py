from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw


def format_overlay_lines(overlay_state: Dict[str, Any]) -> List[str]:
    return [
        f"exp: {overlay_state['experiment_name']}",
        f"step: {overlay_state['checkpoint_step']}",
        f"episode: {overlay_state['episode']}",
        f"env_step: {overlay_state['env_step']}",
        f"action: {overlay_state['action']}",
        f"reward: {overlay_state['instant_reward']:.4f}",
        f"return: {overlay_state['episode_return']:.4f}",
        f"agent_hp: {overlay_state['agent_hp']}",
        f"enemy_hp: {overlay_state['enemy_hp']}",
        f"result: {overlay_state['result']}",
    ]


def draw_overlay(frame: np.ndarray, overlay_state: Dict[str, Any]) -> np.ndarray:
    image = Image.fromarray(_ensure_rgb(frame))
    draw = ImageDraw.Draw(image)
    lines = format_overlay_lines(overlay_state)

    padding = 8
    line_height = 14
    box_width = 360
    box_height = padding * 2 + line_height * len(lines)
    draw.rectangle((6, 6, 6 + box_width, 6 + box_height), fill=(0, 0, 0))

    y = 6 + padding
    for line in lines:
        draw.text((12, y), line, fill=(255, 255, 255))
        y += line_height

    return np.asarray(image, dtype=np.uint8)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[:, :, :3]
    raise ValueError("Frame must be 2D grayscale or 3-channel RGB.")
