from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class OverlayConfig:
    state_color: Tuple[int, int, int] = (255, 255, 255)
    flow_bbox_color: Tuple[int, int, int] = (255, 0, 0)
    person_bbox_color: Tuple[int, int, int] = (0, 255, 0)
    iou_match_color: Tuple[int, int, int] = (0, 255, 255)
    alarm_border_color: Tuple[int, int, int] = (0, 0, 255)
    alarm_border_thickness: int = 5
    
    font_scale: float = 0.6
    font_thickness: int = 2
    line_thickness: int = 2
    
    state_text_offset: Tuple[int, int] = (10, 30)
    info_text_offset: Tuple[int, int] = (10, 60)
