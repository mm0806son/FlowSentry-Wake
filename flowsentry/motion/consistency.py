from __future__ import annotations


class MotionConsistencyCounter:
    def __init__(self, threshold_frames: int) -> None:
        if threshold_frames < 1:
            raise ValueError("threshold_frames must be >= 1")
        self.threshold_frames = threshold_frames
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0

    def update(self, is_consistent: bool) -> bool:
        if is_consistent:
            self._count += 1
        else:
            self._count = 0
        return self._count >= self.threshold_frames
