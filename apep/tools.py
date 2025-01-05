
import numpy as np


def global_to_local(poses, pose0):
    # Unpack global positions and orientations
    x_world, y_world, theta_world = poses[..., 0], poses[..., 1], poses[..., 2]
    x0, y0, theta0 = pose0[..., 0], pose0[..., 1], pose0[..., 2]

    # Calculate the local coordinates
    x_local = (x_world - x0) * np.cos(theta0) + (y_world - y0) * np.sin(theta0)
    y_local = -(x_world - x0) * np.sin(theta0) + (y_world - y0) * np.cos(theta0)

    # Calculate delta theta, normalizing to [-pi, pi]
    delta_theta = (theta_world - theta0 + np.pi) % (2 * np.pi) - np.pi

    return np.stack([x_local, y_local, delta_theta], axis=-1)
