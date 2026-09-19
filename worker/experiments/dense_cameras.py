"""Camera export helpers; world and camera bases are explicitly OpenGL."""
import numpy as np


def viewer_camera(pose, batch_scale, batch_rotation, batch_translation,
                  reference_rotation, reference_translation, world_scale):
    flip = np.diag([1., -1., -1.])
    center = batch_scale * batch_rotation @ pose[:3, 3] + batch_translation
    matrix = np.eye(4)
    matrix[:3, :3] = flip @ reference_rotation @ batch_rotation @ pose[:3, :3] @ flip
    matrix[:3, 3] = flip @ (reference_rotation @ center + reference_translation) * world_scale
    return matrix


def fit_ray_intrinsics(rays, valid, source_size):
    """Fit a pinhole K to learned rays; retain residual because rays need not be pinhole."""
    h, w = rays.shape[:2]
    y, x = np.mgrid[:h, :w]
    good = valid & np.isfinite(rays).all(-1) & (rays[..., 2] > 1e-6)
    good[1::2] = False
    good[:, 1::2] = False
    xy = rays[good, :2] / rays[good, 2:3]
    pixels = np.stack([x[good] + .5, y[good] + .5], axis=1)
    if len(xy) < 100:
        raise ValueError('Insufficient valid rays for intrinsic fit')
    K = np.eye(3)
    errors = []
    for axis in range(2):
        design = np.column_stack([xy[:, axis], np.ones(len(xy))])
        focal, principal = np.linalg.lstsq(design, pixels[:, axis], rcond=None)[0]
        if focal <= 0:
            raise ValueError('Nonpositive fitted focal length')
        K[axis, axis], K[axis, 2] = focal, principal
        errors.append(design @ [focal, principal] - pixels[:, axis])
    source_K = np.diag([source_size[0] / w, source_size[1] / h, 1.]) @ K
    return dict(intrinsics=K.tolist(), source_intrinsics=source_K.tolist(),
                image_size=[w, h], source_image_size=list(source_size),
                intrinsic_fit_pixel_rmse=float(np.sqrt(np.mean(np.sum(np.square(errors), axis=0)))),
                intrinsics_method='Least-squares zero-skew pinhole fit to predicted rays; pixel centers at x+0.5,y+0.5. Source K undoes the two resize operations. This is estimated, not supplied calibration.')
