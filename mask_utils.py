"""Mask 像素级几何工具

给 behavior_rules 用:替代 bbox IoU 的粗糙判定。

核心函数:
- mask_to_binary:把多边形点转成 0/1 二值图(在给定画布尺寸内)
- mask_overlap_ratio:计算两个 mask 的交集占比
- mask_head_region:提取动物 mask 的上 30% 作为"头部"

设计思路:
- 用 OpenCV fillPoly 把多边形栅格化为 mask
- 用 bitwise_and 求交集
- 交集像素数 / 头部 mask 像素数 = "触碰"占比
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


def polygon_to_mask(polygon_pts, h, w) -> np.ndarray:
    """把多边形顶点转成 uint8 二值 mask(h × w)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon_pts is None or len(polygon_pts) < 3:
        return mask
    pts = np.asarray(polygon_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_area(polygon_pts) -> int:
    """多边形面积(像素级近似)"""
    if polygon_pts is None or len(polygon_pts) < 3:
        return 0
    pts = np.asarray(polygon_pts).astype(np.int32)
    return int(cv2.contourArea(pts))


def mask_overlap_area(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """两个 mask 的交集像素数"""
    if mask_a is None or mask_b is None:
        return 0
    if mask_a.shape != mask_b.shape:
        return 0
    inter = cv2.bitwise_and(mask_a, mask_b)
    return int(np.sum(inter > 0))


def mask_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray,
                        by_smaller: bool = True) -> float:
    """交集占较小 mask 的比例(0-1)

    by_smaller=True: overlap / min(area_a, area_b)
    by_smaller=False: overlap / union(A, B) (IoU)
    """
    if mask_a is None or mask_b is None:
        return 0.0
    inter = mask_overlap_area(mask_a, mask_b)
    if inter == 0:
        return 0.0
    if by_smaller:
        denom = min(int(np.sum(mask_a > 0)), int(np.sum(mask_b > 0)))
    else:
        union = cv2.bitwise_or(mask_a, mask_b)
        denom = int(np.sum(union > 0))
    return inter / denom if denom > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """从二值 mask 求最小外接矩形 (x1, y1, x2, y2)"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def head_region_mask(animal_mask: np.ndarray,
                     top_ratio: float = 0.3) -> np.ndarray:
    """从动物 mask 提取上 top_ratio 部分作为"头部区域"

    简单粗糙:mask 竖直 y 方向的最上 30% 高度里的所有像素
    """
    if animal_mask is None:
        return None
    box = mask_bbox(animal_mask)
    if box is None:
        return np.zeros_like(animal_mask)
    x1, y1, x2, y2 = box
    h = y2 - y1 + 1
    cutoff_y = y1 + int(h * top_ratio)
    head = animal_mask.copy()
    head[cutoff_y + 1:, :] = 0
    return head


def visualize_overlap(frame: np.ndarray, head_mask: np.ndarray,
                      bowl_mask: np.ndarray,
                      trigger_color=(0, 100, 255),
                      alpha: float = 0.5) -> np.ndarray:
    """在 frame 上高亮显示头部与盆的重叠区域"""
    if head_mask is None or bowl_mask is None:
        return frame
    inter = cv2.bitwise_and(head_mask, bowl_mask)
    if np.sum(inter) == 0:
        return frame
    overlay = frame.copy()
    overlay[inter > 0] = trigger_color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
