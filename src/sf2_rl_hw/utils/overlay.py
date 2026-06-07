from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def format_overlay_lines(overlay_state: Dict[str, Any]) -> List[str]:
    return [
        f"exp: {overlay_state['experiment_name']}",
        f"step: {overlay_state['checkpoint_step']}",
        f"env_step: {overlay_state['env_step']}",
        f"action: {overlay_state['action']}",
        f"reward: {overlay_state['instant_reward']:.4f}",
        f"return: {overlay_state['episode_return']:.4f}",
        f"agent_hp: {overlay_state['agent_hp']}",
        f"enemy_hp: {overlay_state['enemy_hp']}",
        f"result: {overlay_state['result']}",
    ]


def draw_overlay(frame: np.ndarray, overlay_state: Dict[str, Any]) -> np.ndarray:
    image = Image.fromarray(_ensure_rgb(frame)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    lines = format_overlay_lines(overlay_state)

    margin = 6
    padding_x = 6
    padding_y = 5
    line_spacing = 2
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max((bbox[2] - bbox[0]) for bbox in line_boxes)
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    text_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    box_width = min(frame.shape[1] - margin * 2, text_width + padding_x * 2)
    box_height = min(frame.shape[0] - margin * 2, text_height + padding_y * 2)

    draw.rounded_rectangle(
        (margin, margin, margin + box_width, margin + box_height),
        radius=6,
        fill=(0, 0, 0, 144),
    )

    y = margin + padding_y
    for line in lines:
        draw.text((margin + padding_x, y), line, font=font, fill=(255, 255, 255, 255))
        _, _, _, bottom = draw.textbbox((margin + padding_x, y), line, font=font)
        y = bottom + line_spacing

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[:, :, :3]
    raise ValueError("Frame must be 2D grayscale or 3-channel RGB.")
