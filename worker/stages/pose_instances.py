"""One-to-one support for pose detections from independent person instances."""

import numpy as np
from scipy.optimize import linear_sum_assignment

# Preserve the existing strict support threshold used by track_people.
MIN_INSTANCE_IOU = 0.2


def match_pose_instances(pose_boxes, instance_boxes):
    """Return {pose index: (instance index, IoU)} for supported detections.

    Maximize the number of supported pairs, then their total IoU. Each instance can
    support only one pose. Dummy columns leave unsupported/duplicate poses unmatched;
    no rectangle mask is evidence of a person. Empty inputs produce no matches.
    """
    poses = np.asarray(pose_boxes, dtype=float).reshape(-1, 4)
    instances = np.asarray(instance_boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(poses).all() or not np.isfinite(instances).all():
        raise ValueError("pose and instance boxes must be finite")
    if not len(poses) or not len(instances):
        return {}
    low = np.maximum(poses[:, None, :2], instances[None, :, :2])
    high = np.minimum(poses[:, None, 2:], instances[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    pose_area = np.maximum(poses[:, 2:] - poses[:, :2], 0).prod(axis=1)
    instance_area = np.maximum(instances[:, 2:] - instances[:, :2], 0).prod(axis=1)
    union = pose_area[:, None] + instance_area[None, :] - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    supported = overlap > MIN_INSTANCE_IOU
    # Losing one supported pair cannot be offset by improvements to all other IoUs.
    reward = np.zeros((len(poses), len(instances) + len(poses)))
    reward[:, : len(instances)] = np.where(
        supported, min(len(poses), len(instances)) + 1 + overlap, 0
    )
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return {
        int(row): (int(col), float(overlap[row, col]))
        for row, col in zip(rows, cols)
        if col < len(instances) and supported[row, col]
    }
