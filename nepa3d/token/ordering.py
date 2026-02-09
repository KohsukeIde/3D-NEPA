import numpy as np


def morton3d(xyz, bits=10):
    grid = np.clip((xyz * 0.5 + 0.5) * (2**bits - 1), 0, 2**bits - 1).astype(
        np.uint32
    )
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    def split_by_2(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x30000FF
        v = (v | (v << 8)) & 0x300F00F
        v = (v | (v << 4)) & 0x30C30C3
        v = (v | (v << 2)) & 0x9249249
        return v

    xx = split_by_2(x)
    yy = split_by_2(y) << 1
    zz = split_by_2(z) << 2
    return (xx | yy | zz).astype(np.uint32)


def sort_by_ray_direction(ray_d):
    d = ray_d / (np.linalg.norm(ray_d, axis=1, keepdims=True) + 1e-9)
    theta = np.arctan2(d[:, 1], d[:, 0])
    phi = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    idx = np.lexsort((phi, theta))
    return idx
