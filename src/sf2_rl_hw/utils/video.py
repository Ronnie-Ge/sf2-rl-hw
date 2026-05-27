from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


class VideoFrameWriter:
    def __init__(self, output_parent: Path, fps: int) -> None:
        self.output_parent = output_parent
        self.output_parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames_dir = Path(tempfile.mkdtemp(prefix="frames-", dir=str(output_parent)))
        self.frame_count = 0

    def add_frame(self, frame: np.ndarray) -> None:
        frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        Image.fromarray(frame).save(frame_path)
        self.frame_count += 1

    def finalize(self, output_path: Path) -> Optional[Path]:
        if self.frame_count == 0:
            raise RuntimeError("No frames were captured for video output.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_log = output_path.with_suffix(".ffmpeg.log")
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            str(self.frames_dir / "frame_%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            error_log.write_text(f"ffmpeg executable not found: {exc}\n", encoding="utf-8")
            raise RuntimeError(
                f"ffmpeg executable not found for {output_path}. Frame dump kept at {self.frames_dir}. "
                f"See {error_log} for details."
            ) from exc

        if result.returncode != 0:
            error_log.write_text(result.stderr, encoding="utf-8")
            raise RuntimeError(
                f"ffmpeg failed for {output_path}. Frame dump kept at {self.frames_dir}. "
                f"See {error_log} for details."
            )

        shutil.rmtree(self.frames_dir, ignore_errors=True)
        return output_path
